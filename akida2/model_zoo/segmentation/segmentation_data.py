#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Data loading for the segmentation example (Cityscapes).

Images are cropped/resized to 384x384 tiles for training; evaluation instead
runs a sliding window over the full-resolution image (see
segmentation_eval.py) since mIoU is measured against the original mask
resolution. Labels use the standard 19-class Cityscapes "trainId" scheme
(the 34 raw label IDs collapse to 19 classes + an ignored/void class).

Two input conventions exist side by side in this pipeline (documented here
so eval/benchmark don't look inconsistent):
  - The PyTorch/ONNX float and quantized-pre-conversion models take
    ImageNet-normalized float32 NCHW tensors (mean/std below).
  - The Akida-converted model takes RAW UINT8 NHWC tiles -- the
    normalization is baked into the quantized graph itself (the quantizeml
    ONNX quantization step captures it), the same "rescaling folded into
    the model" pattern used by this zoo's Keras-based examples (e.g. vww's
    `insert_rescaling`). `get_samples()` below returns uint8, matching the
    zoo-wide convention and what quantizeml/cnn2snn/benchmark actually need.
"""

import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

TILE_SIZE = 384
OVERLAP = 64
DOWNSAMPLE = 0.5
NUM_CLASSES = 19
IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle',
]

# Cityscapes raw labelId -> 19-class trainId (everything else maps to IGNORE_INDEX)
ID_TO_TRAINID = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
                 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _id, _train_id in ID_TO_TRAINID.items():
    LUT[_id] = _train_id

_NORM_TF = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def _image_mask_paths(data_path, split):
    img_dir = os.path.join(data_path, 'leftImg8bit', split)
    mask_dir = os.path.join(data_path, 'gtFine', split)
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*', '*_leftImg8bit.png')))
    mask_paths = []
    for p in img_paths:
        city = os.path.basename(os.path.dirname(p))
        fname = os.path.basename(p).replace('leftImg8bit', 'gtFine_labelIds')
        mask_paths.append(os.path.join(mask_dir, city, fname))
    return img_paths, mask_paths


class CityscapesHalfResCrops(Dataset):
    """Random 384x384 crops of the half-resolution ("DOWNSAMPLE=0.5") image,
    for training. For "val"/"test" splits, returns the full half-resolution
    image uncropped (the sliding-window eval tiles it at full resolution
    instead -- see segmentation_eval.py)."""

    def __init__(self, data_path, split='train'):
        self.img_paths, self.mask_paths = _image_mask_paths(data_path, split)
        self.split = split
        if split == 'train':
            self.tf = A.Compose([
                A.RandomCrop(TILE_SIZE, TILE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            self.tf = _NORM_TF

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = LUT[cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)]
        if DOWNSAMPLE != 1.0:
            img = cv2.resize(img, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
            mask = cv2.resize(mask, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE, interpolation=cv2.INTER_NEAREST)
        aug = self.tf(image=img, mask=mask)
        return aug['image'], torch.as_tensor(aug['mask'], dtype=torch.long)


def get_data(data_path, batch_size=4, num_workers=2):
    """Loads the Cityscapes train/val splits.

    Args:
        data_path (str): Cityscapes root (contains leftImg8bit/ and gtFine/).
        batch_size (int): training batch size.
        num_workers (int): DataLoader worker count.

    Returns:
        (train_loader, val_dataset): a DataLoader for training (random
        384x384 crops), and the raw val Dataset (for sliding-window
        full-image evaluation -- see segmentation_eval.py, which needs the
        original image paths, not a cropped/batched loader).
    """
    train_ds = CityscapesHalfResCrops(data_path, 'train')
    val_ds = CityscapesHalfResCrops(data_path, 'val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_ds


def get_samples(data_path, input_shape=(TILE_SIZE, TILE_SIZE, 3), num_samples=1024, seed=0, split='train'):
    """Loads raw (uint8, unnormalized) 384x384 RGB tiles for calibration and
    hardware benchmarking.

    Args:
        data_path (str): Cityscapes root.
        input_shape (tuple): (height, width, channels) of each tile.
        num_samples (int): number of tiles to return.
        seed (int): RNG seed.
        split (str): which split to sample tiles from.

    Returns:
        np.ndarray: (num_samples, height, width, channels), dtype uint8.
    """
    img_paths, _ = _image_mask_paths(data_path, split)
    rng = np.random.default_rng(seed)
    h, w = input_shape[:2]
    samples = np.zeros((num_samples, h, w, 3), dtype=np.uint8)
    idx = rng.choice(len(img_paths), size=num_samples, replace=True)
    for i, path_idx in enumerate(idx):
        img = cv2.cvtColor(cv2.imread(img_paths[path_idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
        y0 = rng.integers(0, max(img.shape[0] - h, 1))
        x0 = rng.integers(0, max(img.shape[1] - w, 1))
        samples[i] = img[y0:y0 + h, x0:x0 + w]
    return samples
