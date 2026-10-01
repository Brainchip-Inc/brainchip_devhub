#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Segmentation evaluation for PyTorch (.pth), ONNX (.onnx), or Akida (.fbz)
models.

Reports mean IoU (mIoU) over the Cityscapes val set -- this is a
segmentation task, not classification, so there is no "accuracy" metric
here. Full-resolution images are evaluated with a sliding-window tiling
reconstruction (the model only ever sees 384x384 tiles), in two modes:

  - "mosaic": each tile's logits simply overwrite that region -- fast, but
    has visible seams at tile boundaries.
  - "hann": tiles overlap and are blended with a 2D Hann window, removing
    seams at the cost of ~4x the tile inferences per image (overlap=64 on a
    384-tile means a large fraction of pixels are covered 2-4 times).

Both modes are reported for the PyTorch/ONNX (float/pre-conversion) models.
The Akida-converted model is only evaluated in mosaic mode here, matching
the project's own practice: the Akida software backend is CPU-bound in this
environment, and hann mode's ~4x tile count makes a full 500-image val pass
impractical (mosaic-only already takes over an hour -- see README.md).

Example
-------
    python segmentation_eval.py -d ./data/cityscapes -l pretrained_models/unetv4akida_segmentation.pth
"""
import argparse
import json
import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from segmentation_data import (LUT, CLASS_NAMES, NUM_CLASSES, TILE_SIZE, OVERLAP, DOWNSAMPLE,
                                 _image_mask_paths, _NORM_TF)
from segmentation_model import build_segmentation_model


def get_tile_starts(size, tile, overlap):
    stride = tile - overlap
    starts = list(range(0, size - tile, stride))
    if not starts or starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def _make_hann_window(h, w):
    wy = torch.hann_window(h, periodic=False)
    wx = torch.hann_window(w, periodic=False)
    return torch.outer(wy, wx).clamp_min(1e-3)[None, None]


@torch.no_grad()
def predict_tiled(predict_fn, img, tile=TILE_SIZE, overlap=OVERLAP, use_hann=False):
    """Tiles `img` (1, 3, H, W) and runs `predict_fn(patch) -> (1, C, tile, tile)
    logits` over every tile, reassembling full (1, C, H, W) logits.
    `use_hann` blends overlapping tiles with a Hann window; otherwise each
    tile's region is simply overwritten (mosaic)."""
    _, _, h, w = img.shape
    ys = get_tile_starts(h, tile, overlap)
    xs = get_tile_starts(w, tile, overlap)

    acc = torch.zeros((1, NUM_CLASSES, h, w))
    wgt = torch.zeros((1, 1, h, w)) if use_hann else None
    win = _make_hann_window(tile, tile) if use_hann else None

    for y in ys:
        for x in xs:
            patch = img[:, :, y:y + tile, x:x + tile]
            logits = predict_fn(patch)
            if use_hann:
                acc[:, :, y:y + tile, x:x + tile] += logits * win
                wgt[:, :, y:y + tile, x:x + tile] += win
            else:
                acc[:, :, y:y + tile, x:x + tile] = logits
    return acc / wgt.clamp_min(1e-6) if use_hann else acc


def eval_cityscapes_set(predict_fn, img_paths, mask_paths, use_hann, max_images=None,
                         akida_uint8=False):
    """Confusion-matrix-based per-class + mean IoU over a Cityscapes split.

    Args:
        predict_fn: callable(patch) -> (1, NUM_CLASSES, tile, tile) logits.
        akida_uint8: True for the Akida model, whose input is raw uint8
            NHWC (not ImageNet-normalized float NCHW like the PyTorch/ONNX
            models -- see segmentation_data.py module docstring).

    Returns:
        (iou, miou, gt_percent): per-class IoU array, mean IoU, and each
        class's share of ground-truth pixels (for the printed table).
    """
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    gt_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    n = len(img_paths) if max_images is None else min(max_images, len(img_paths))

    for i in tqdm(range(n), desc='Evaluating'):
        img_full = cv2.cvtColor(cv2.imread(img_paths[i]), cv2.COLOR_BGR2RGB)
        mask_full = cv2.imread(mask_paths[i], cv2.IMREAD_UNCHANGED)
        gt = LUT[mask_full]
        h, w = gt.shape

        img_05 = cv2.resize(img_full, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE, interpolation=cv2.INTER_LINEAR)
        if akida_uint8:
            img_t = torch.from_numpy(img_05).unsqueeze(0).permute(0, 3, 1, 2)  # raw uint8, NCHW for tiling
        else:
            img_t = _NORM_TF(image=img_05)['image'].unsqueeze(0)

        logits_05 = predict_tiled(predict_fn, img_t, use_hann=use_hann)
        pred = F.interpolate(logits_05, size=(h, w), mode='bilinear', align_corners=False) \
            .argmax(1).squeeze(0).numpy().astype(np.int64)

        valid = gt != 255
        g, p = gt[valid].astype(np.int64), pred[valid]
        gt_pixel_counts += np.bincount(g, minlength=NUM_CLASSES)
        conf += np.bincount(NUM_CLASSES * g + p, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

    inter = np.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    iou = np.where(union > 0, inter / union, np.nan)
    gt_percent = gt_pixel_counts / max(gt_pixel_counts.sum(), 1) * 100.0
    return iou, float(np.nanmean(iou)), gt_percent


def _print_iou_table(iou, miou, gt_percent, label):
    print(f'\nPer-class IoU ({label}):')
    print(f"{'Class':<18} | {'IoU':>8} | {'GT %':>7}")
    print('-' * 40)
    for name, i, gp in zip(CLASS_NAMES, iou, gt_percent):
        i_str = f'{i * 100:6.2f}%' if np.isfinite(i) else '   NaN'
        print(f'{name:<18} | {i_str:>8} | {gp:6.2f}%')
    print('-' * 40)
    print(f"{'mIoU':<18} | {miou * 100:6.2f}%")


def _build_predict_fn(loadmodel):
    """Returns (predict_fn, akida_uint8, modes, metric_prefix, extra_metrics_fn).

    modes: list of (use_hann, metrics_key_suffix) pairs to evaluate and report.
    extra_metrics_fn(metrics_dict): called once (not per mode) to add any
    metrics that don't vary by tiling mode (e.g. param count).
    """
    if loadmodel.endswith('.fbz'):
        import akida
        model = akida.Model(loadmodel)

        def predict_fn(patch):
            out = model.predict(patch.permute(0, 2, 3, 1).numpy().astype('uint8'))
            return torch.from_numpy(out).permute(0, 3, 1, 2)

        # Mosaic only -- see module docstring for why.
        return predict_fn, True, [(False, '_mosaic')], 'w8a8_akida_miou', None

    if loadmodel.endswith('.onnx'):
        import onnxruntime as ort
        session = ort.InferenceSession(loadmodel)
        input_name = session.get_inputs()[0].name

        def predict_fn(patch):
            out = session.run(None, {input_name: patch.numpy().astype(np.float32)})[0]
            return torch.from_numpy(out)

        return predict_fn, False, [(False, '_mosaic'), (True, '_hann')], 'onnx_miou', None

    model = build_segmentation_model()
    model.load_state_dict(torch.load(loadmodel, map_location='cpu'))
    model.eval()

    def predict_fn(patch):
        with torch.no_grad():
            return model(patch)

    def extra_metrics(metrics):
        metrics['params'] = f'{sum(p.numel() for p in model.parameters()):,}'

    return predict_fn, False, [(False, '_mosaic'), (True, '_hann')], 'float_miou', extra_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.pth PyTorch, .onnx, or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/cityscapes',
                        help='Cityscapes root (contains leftImg8bit/ and gtFine/)')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--max-images', type=int, default=None,
                        help='Subsample the eval set (full val split is 500 images; the Akida '
                             'software backend takes ~1h for mosaic-only on all 500)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write mIoU (and param count for the PyTorch .pth) to docs/metrics.json')
    args = parser.parse_args()

    img_paths, mask_paths = _image_mask_paths(args.data, args.split)
    predict_fn, akida_uint8, modes, metric_prefix, extra_metrics_fn = _build_predict_fn(args.loadmodel)

    metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
    metrics = json.loads(metrics_path.read_text()) if (args.save_metrics and metrics_path.exists()) else {}

    for use_hann, suffix in modes:
        iou, miou, gt_pct = eval_cityscapes_set(
            predict_fn, img_paths, mask_paths, use_hann=use_hann,
            max_images=args.max_images, akida_uint8=akida_uint8)
        _print_iou_table(iou, miou, gt_pct, f"{metric_prefix}, {'hann' if use_hann else 'mosaic'}")
        if args.save_metrics:
            metrics[f'{metric_prefix}{suffix}'] = f'{miou * 100:.2f}%'

    if args.save_metrics:
        if extra_metrics_fn is not None:
            extra_metrics_fn(metrics)
        if akida_uint8:
            metrics['n_eval_images'] = str(args.max_images or len(img_paths))
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
