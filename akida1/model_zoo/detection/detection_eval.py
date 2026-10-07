#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Detection evaluation (mAP) for tf_keras or akida YOLOv2 models.

Example
-------
    python detection_eval.py -d /data/voc/ -l yolo_akidanet_detection.h5
"""
import argparse
import json
import pathlib

import akida
import tensorflow as tf

from tf_keras import Model
from tf_keras.layers import Reshape
# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()

from collections import defaultdict

import numpy as np
import tf_keras as keras
from tf_keras.src.utils import io_utils
from tqdm import tqdm

from cnn2snn import load_quantized_model

from akida_models.detection.box_utils import compute_overlap
from akida_models.detection.data_utils import Coord
from akida_models.detection.processing import (BoundingBox, desize_bboxes,
                                               get_affine_transform, preprocess_image)

from detection_data import LABELS, get_anchors, get_voc_dataset
from brainchip_utils.hardware_utils import get_akida_device


# ---------------------------------------------------------------------------
# Local copies of the akida_models (1.14.0) YOLO decoding and mAP evaluation
# ---------------------------------------------------------------------------
# `decode_output` is copied from akida_models/detection/processing.py
# (Copyright 2020 Brainchip Holdings Ltd., Apache 2.0) and `MapEvaluation`
# from akida_models/detection/map_evaluation.py (Copyright 2017-2018 Fizyr,
# Apache 2.0, adapted by Brainchip). Both are copied here unchanged, so that
# local modifications can be offered back upstream as a clean diff.


def decode_output(output, anchors, nb_classes, obj_threshold=0.5, nms_threshold=0.5):
    """ Decodes a YOLO model output.

    Args:
        output (tf.Tensor): model output to decode
        anchors (list): list of anchors boxes
        nb_classes (int): number of classes
        obj_threshold (float, optional): confidence threshold for a box. Defaults to 0.5.
        nms_threshold (float, optional): non-maximal suppression threshold. Defaults to 0.5.

    Returns:
        List of `BoundingBox` objects
    """

    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def _softmax(x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x / np.min(x) * t

        e_x = np.exp(x)

        return e_x / e_x.sum(axis, keepdims=True)

    grid_h, grid_w, nb_box = output.shape[:3]

    boxes = []

    # decode the output by the network
    output[..., 4] = _sigmoid(output[..., 4])
    output[..., 5:] = output[..., 4][..., np.newaxis] * _softmax(output[..., 5:])
    output[..., 5:] *= output[..., 5:] > obj_threshold

    col, row, _ = np.meshgrid(np.arange(grid_w), np.arange(grid_h), np.arange(nb_box))

    x = (col + _sigmoid(output[..., 0])) / grid_w
    y = (row + _sigmoid(output[..., 1])) / grid_h
    w = np.array(anchors)[:, 0] * np.exp(output[..., 2]) / grid_w
    h = np.array(anchors)[:, 1] * np.exp(output[..., 3]) / grid_h

    x1 = np.maximum(x - w / 2, 0)
    y1 = np.maximum(y - h / 2, 0)
    x2 = np.minimum(x + w / 2, grid_w)
    y2 = np.minimum(y + h / 2, grid_h)

    confidence = output[..., 4]
    classes = output[..., 5:]
    mask = np.sum(classes, axis=-1) > 0
    indices = np.where(mask)

    for i in range(len(indices[0])):
        row_idx, col_idx, box_idx = indices[0][i], indices[1][i], indices[2][i]

        box = BoundingBox(x1[row_idx, col_idx, box_idx],
                          y1[row_idx, col_idx, box_idx],
                          x2[row_idx, col_idx, box_idx],
                          y2[row_idx, col_idx, box_idx],
                          confidence[row_idx, col_idx, box_idx],
                          classes[row_idx, col_idx, box_idx])

        boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_classes):
        sorted_indices = np.argsort([box.classes[c] for box in boxes])[::-1]
        for ind, index_i in enumerate(sorted_indices):
            if boxes[index_i].score == 0 or boxes[index_i].classes[c] == 0:
                continue

            for j in range(ind + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if boxes[index_j].score == 0:
                    continue

                # filter out redundant boxes (same class and overlapping too
                # much)
                if (boxes[index_i].iou(boxes[index_j]) >= nms_threshold) and (
                        c == boxes[index_i].get_label()) and (
                            c == boxes[index_j].get_label()):
                    boxes[index_j].score = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


class MapEvaluation(keras.callbacks.Callback):
    """ Evaluate a given dataset using a given model.
        Code originally from https://github.com/fizyr/keras-retinanet.
        Note that mAP is computed for IoU thresholds from 0.5 to 0.95
        with a step size of 0.05.

        Args:
            model (keras.Model): model to evaluate.
            val_data (dict): dictionary containing validation data as obtained
             using `preprocess_widerface.py` module
            num_valid (int): the length of the validation dataset
            labels (list): list of labels as strings
            anchors (list): list of anchors boxes
            period (int, optional): periodicity the precision is printed,
                defaults to once per epoch. Defaults to 1.
            obj_threshold (float, optional): confidence threshold for a box. Defaults to 0.5.
            nms_threshold (float, optional): non-maximal suppression threshold. Defaults to 0.5.
            max_box_per_image (int, optional): maximum number of detections per
                image, Defaults to 10.
            preserve_aspect_ratio (bool, optional): Whether aspect ratio is preserved
                during resizing. Defaults to False.
            is_keras_model (bool, optional): indicated if the model is a Keras
                model (True) or an Akida model (False). Defaults to True.
            decode_output_fn (Callable, optional): function to decode model's outputs.
                Defaults to :func:`decode_output` (yolo decode output function).

        Returns:
            A dict mapping class names to mAP scores.
    """

    def __init__(self,
                 model,
                 val_data,
                 num_valid,
                 labels,
                 anchors,
                 period=1,
                 obj_threshold=0.5,
                 nms_threshold=0.5,
                 max_box_per_image=10,
                 preserve_aspect_ratio=False,
                 is_keras_model=True,
                 decode_output_fn=decode_output):

        super().__init__()
        self._model = model
        self._data = val_data
        self._data_len = num_valid
        self._labels = labels
        self._num_classes = len(labels)
        self._anchors = anchors
        self._period = period
        self._obj_threshold = obj_threshold
        self._nms_threshold = nms_threshold
        self._max_box_per_image = max_box_per_image
        self._preserve_aspect_ratio = preserve_aspect_ratio
        self._is_keras_model = is_keras_model
        self._decode_output = decode_output_fn

    def on_epoch_end(self, epoch, logs=None):
        """ Keras callback called at the end of an epoch.

        Args:
            epoch (int): index of epoch.
            logs (dict, optional): metric results for this training epoch, and
                for the validation epoch if validation is performed. Validation
                result keys are prefixed with val. For training epoch, the
                values of the Model’s metrics are returned.
                Example: {‘loss’: 0.2, ‘acc’: 0.7}. Defaults to None.
        """
        epoch += 1
        if self._period != 0 and (epoch % self._period == 0 or
                                  epoch == self.params.get('epochs', -1)):
            _map_dict, average_precisions = self.evaluate_map()
            mean_map = sum(_map_dict.values()) / len(_map_dict)
            io_utils.print_msg("")
            io_utils.print_msg('mAP 50: {:.4f}'.format(_map_dict[0.5]))
            io_utils.print_msg('mAP 75: {:.4f}\n'.format(_map_dict[0.75]))
            for label, average_precision in average_precisions.items():
                io_utils.print_msg(self._labels[label] + ' {:.4f}'.format(average_precision))
            io_utils.print_msg('mAP: {:.4f}\n'.format(mean_map))
            logs.update({'map': mean_map})

    def evaluate_map(self):
        """ Evaluates current mAP score on the model. mAP is computed for IoU
        thresholds from 0.5 to 0.95 with a step size of 0.05

        Returns:
            tuple: a dictionary containing mAP for each threshold and a dictionary of label
            containing mAP for each class.
        """
        # predictions, overlaps and 10 IoU thresholds
        self._pbar = tqdm(total=self._data_len + self._num_classes + 10, leave=False)

        all_detections, all_annotations = self._get_predictions()
        all_overlaps = self._compute_all_overlaps(all_detections, all_annotations)

        # Thresholds from 0.5 to 0.95 with a step size of 0.05
        iou_thresholds = np.linspace(0.5, 0.95, num=10)
        total_iterations = len(iou_thresholds)
        mean_avgs = defaultdict(float)
        map_dict = {}

        for th in iou_thresholds:
            self._pbar.set_description(f"Computing average precisions th = {th:.2f}")
            average_precisions = self._calc_avg_precisions(all_detections, all_annotations,
                                                           all_overlaps, th)
            _map = sum(average_precisions.values()) / len(average_precisions)
            map_dict[th] = _map

            for label, ap in average_precisions.items():
                mean_avgs[label] += ap / total_iterations
            self._pbar.update(1)
        self._pbar.close()
        keras.backend.clear_session()

        return map_dict, mean_avgs

    def _load_annotations(self, data):
        objects = data['objects']
        h, w, _ = data['image'].shape
        bbox = objects['bbox'].numpy()
        labels = objects['label'].numpy()

        x1 = (bbox[:, Coord.x1] * w).astype(int)
        y1 = (bbox[:, Coord.y1] * h).astype(int)
        x2 = (bbox[:, Coord.x2] * w).astype(int)
        y2 = (bbox[:, Coord.y2] * h).astype(int)

        return np.column_stack([x1, y1, x2, y2, labels])

    def _get_predictions(self):
        # gather all detections and annotations
        all_detections = [[None
                           for _ in range(self._num_classes)]
                          for _ in range(self._data_len)]
        all_annotations = [[None
                            for _ in range(self._num_classes)]
                           for _ in range(self._data_len)]
        self._pbar.set_description("Getting predictions")

        for i, data in enumerate(self._data):
            raw_image = data['image']
            raw_height, raw_width, _ = raw_image.shape
            input_shape = self._model.input_shape[1:] if self._is_keras_model \
                else self._model.input_shape

            affine_transform = None
            if self._preserve_aspect_ratio:
                center = np.array([raw_width / 2., raw_height / 2.], dtype=np.float32)
                affine_transform = get_affine_transform(center, [raw_width, raw_height],
                                                        [input_shape[1],
                                                        input_shape[0]])

            image = preprocess_image(raw_image.numpy(), input_shape, affine_transform)
            input_image = image[np.newaxis, :]

            if self._is_keras_model:
                output = self._model.predict(input_image, verbose=0)[0]
            else:
                potentials = self._model.predict(input_image)[0]

                if self._anchors:
                    h, w, _ = potentials.shape
                    output = potentials.reshape(
                        (h, w, len(self._anchors), 4 + 1 + self._num_classes))
                else:
                    output = potentials

            pred_boxes = self._decode_output(output, self._anchors, self._num_classes,
                                             self._obj_threshold, self._nms_threshold)

            score = np.array([box.get_score() for box in pred_boxes])
            pred_labels = np.array([box.get_label() for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = desize_bboxes(pred_boxes, score, raw_height, raw_width,
                                           input_shape[0], input_shape[1],
                                           self._preserve_aspect_ratio)
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # limit the number of predictions to max_box_per_image based on
            # score
            number_of_predictions = pred_boxes.shape[0]
            if number_of_predictions > self._max_box_per_image:
                pred_labels = pred_labels[:self._max_box_per_image]
                pred_boxes = pred_boxes[:self._max_box_per_image, :]

            # copy detections to all_detections
            for label in range(self._num_classes):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = self._load_annotations(data)

            # copy ground truth to all_annotations
            for label in range(self._num_classes):
                all_annotations[i][label] = annotations[annotations[:, 4] ==
                                                        label, :4].copy()
            self._pbar.update(1)

        return all_detections, all_annotations

    def _compute_all_overlaps(self, all_detections, all_annotations):
        all_overlaps = {}
        self._pbar.set_description("Computing overlaps")

        for label in range(self._num_classes):
            for i in range(self._data_len):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                overlaps = compute_overlap(detections, annotations,
                                           mode="outer_product", box_format="xyxy")
                all_overlaps[(i, label)] = overlaps
            self._pbar.update(1)

        return all_overlaps

    def _calc_avg_precisions(self, all_detections, all_annotations, all_overlaps, iou_threshold):
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(self._num_classes):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(self._data_len):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                overlaps = all_overlaps[(i, label)]
                detected_annotations = []

                for idx, d in enumerate(detections):
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    assigned_annotation = np.argmax(overlaps, axis=1)[idx]
                    max_overlap = overlaps[idx, assigned_annotation]

                    if (max_overlap >= iou_threshold and
                            assigned_annotation not in detected_annotations):
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(
                true_positives + false_positives,
                np.finfo(np.float64).eps)

            # compute average precision
            average_precision = self._compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

    @staticmethod
    def _compute_ap(recall, precision):
        """ Compute the average precision, given the recall and precision
        curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.

        Args:
            recall (list): the recall curve
            precision (list): the precision curve

        Returns:
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/voc',
                        help='VOC dataset root (directory containing the VOC tar archives)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write mAP@0.5 (and param count for .h5) to metrics.json')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    anchors = get_anchors()

    if args.loadmodel.endswith('.h5'):
        model = load_quantized_model(args.loadmodel)
        is_keras_model = True

        # MapEvaluation expects a Keras model whose output is already shaped
        # (grid_h, grid_w, num_anchors, 5+classes); it only reshapes the flat
        # conv output internally for Akida models. Wrap with the same
        # YOLO_output reshape used during training, for evaluation only.
        grid_size = model.output_shape[1:3]
        num_classes = model.output_shape[-1] // len(anchors) - 5
        eval_output = Reshape((grid_size[0], grid_size[1], len(anchors), 5 + num_classes),
                              name='YOLO_output')(model.output)
        eval_model = Model(model.input, eval_output)
    elif args.loadmodel.endswith('.fbz'):
        model = akida.Model(args.loadmodel)
        is_keras_model = False
        eval_model = model

        device = get_akida_device(target_version=model.ip_version)
        if device is not None:
            model.map(device, mode=akida.MapMode.HwPr)
            print('Running inference on Akida hardware device')
            model.summary()

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    val_data, labels, num_valid = get_voc_dataset(args.data, labels=LABELS, training=False)

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    map_evaluator = MapEvaluation(eval_model, val_data, num_valid, labels, anchors,
                                  is_keras_model=is_keras_model)
    # map_dict holds mAP per IoU threshold (0.5 to 0.95). Following the
    # PASCAL VOC convention, only mAP at IoU 0.5 is reported. The per-class
    # APs also returned are averaged over all thresholds, so are not shown.
    map_dict, _ = map_evaluator.evaluate_map()
    map_50 = map_dict[0.5]
    print(f'mAP (IoU 0.5): {map_50:.4f}')

    # ---------------------------------------------------------------------------
    # Persist metrics
    # ---------------------------------------------------------------------------
    if args.save_metrics:
        # This is used to update the stored metrics that are used to generate the
        # performance tables in the README of this folder.
        # This should only be used for code maintenance, when the model or training
        # pipeline is updated and a new trained model integrated.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        map_str = f'{map_50 * 100:.2f}%'
        if not is_keras_model:
            metrics['akida_map'] = map_str
        elif 'qat' in pathlib.Path(args.loadmodel).stem:
            metrics['qat_map'] = map_str
        else:
            metrics['float_map'] = map_str
            metrics['params'] = f'{model.count_params():,}'
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
