#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Segmentation training (PyTorch -- see segmentation_model.py for why this
example diverges from the zoo's usual tf_keras convention).

Weighted cross-entropy (class weights below counter Cityscapes' heavy class
imbalance -- "road" is ~38% of pixels, "train"/"motorcycle" well under 1%),
AdamW + cosine LR schedule, mIoU tracked via torchmetrics. Validation runs a
Hann-windowed sliding-window prediction over a handful of full val images
each epoch (full dataset-wide mIoU, mosaic vs. Hann, both tiling modes, is
reported by segmentation_eval.py instead -- see its module docstring for why
the two modes differ).

Example
-------
    python segmentation_train.py -d ./data/cityscapes -e 100 \\
        -l models/unetv4akida_segmentation_untrained.pth -s models/unetv4akida_segmentation.pth
"""
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassJaccardIndex

from segmentation_model import build_segmentation_model
from segmentation_data import get_data, LUT, DOWNSAMPLE, TILE_SIZE, OVERLAP, NUM_CLASSES, IGNORE_INDEX, _NORM_TF

# Cityscapes class weights (by trainId 0-18), clamped and mean-normalized --
# counters the heavy class imbalance (see module docstring).
CLASS_WEIGHTS = torch.tensor([
    0.5, 2.0, 1.0, 2.0, 2.0, 3.0, 4.0, 3.0, 1.0, 2.0, 1.0,
    3.0, 4.0, 1.0, 2.0, 2.0, 4.0, 4.0, 3.0,
])


def _make_hann_window(h, w, device):
    wy = torch.hann_window(h, periodic=False, device=device)
    wx = torch.hann_window(w, periodic=False, device=device)
    return torch.outer(wy, wx).clamp_min(1e-3)[None, None]


def sliding_window_predict(model, img, num_classes=NUM_CLASSES, tile=(TILE_SIZE, TILE_SIZE),
                            overlap=OVERLAP, device='cuda'):
    """Hann-windowed sliding-window inference over a full (possibly larger
    than one tile) image -- used for the quick per-epoch validation check
    here, and (in its "mosaic" form) by segmentation_eval.py's full
    dataset-wide evaluation."""
    model.eval()
    with torch.no_grad():
        _, _, h, w = img.shape
        th, tw = tile
        sy, sx = th - overlap, tw - overlap
        acc = torch.zeros((1, num_classes, h, w), device=device)
        wgt = torch.zeros((1, 1, h, w), device=device)
        win = _make_hann_window(th, tw, device)
        for y in range(0, max(h - th + 1, 1), sy):
            for x in range(0, max(w - tw + 1, 1), sx):
                y0, x0 = min(y, h - th), min(x, w - tw)
                patch = img[:, :, y0:y0 + th, x0:x0 + tw]
                logits = model(patch)
                acc[:, :, y0:y0 + th, x0:x0 + tw] += logits * win
                wgt[:, :, y0:y0 + th, x0:x0 + tw] += win
        return acc / wgt.clamp_min(1e-6)


def train_segmentation(model, train_loader, val_dataset, epochs, learning_rate,
                        weight_decay=1e-4, device='cuda', seed=42):
    """Trains `model` in place, returns (history, best_state_dict, best_miou)."""
    torch.manual_seed(seed)
    model = model.to(device)
    class_weights = CLASS_WEIGHTS.clamp(max=10)
    class_weights = (class_weights / class_weights.mean()).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    miou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).to(device)

    history = {'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': []}
    best_miou, best_state = 0.0, None

    print('Starting training...')
    for epoch in range(epochs):
        model.train()
        tr_loss = tr_miou = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            tr_miou += miou_metric(logits.argmax(1), masks).item() * imgs.size(0)
        tr_loss /= len(train_loader.dataset)
        tr_miou /= len(train_loader.dataset)
        scheduler.step()

        # Quick per-epoch validation: Hann sliding-window over a handful of
        # full val images (not the whole val set -- see segmentation_eval.py
        # for the full dataset-wide mIoU used to report final numbers).
        model.eval()
        val_loss = val_miou = 0.0
        n_quick_val = min(5, len(val_dataset))
        for i in range(n_quick_val):
            img_path, mask_path = val_dataset.img_paths[i], val_dataset.mask_paths[i]
            img_full = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask_full = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            target_full = torch.from_numpy(LUT[mask_full]).long().unsqueeze(0).to(device)

            img_05 = cv2.resize(img_full, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
            img_05_t = _NORM_TF(image=img_05)['image'].unsqueeze(0).to(device)
            logits_05 = sliding_window_predict(model, img_05_t, device=device)
            logits_full = F.interpolate(logits_05, size=target_full.shape[-2:],
                                         mode='bilinear', align_corners=False)
            val_loss += criterion(logits_full, target_full).item()
            val_miou += miou_metric(logits_full.argmax(1), target_full).item()
        val_loss /= n_quick_val
        val_miou /= n_quick_val

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(tr_miou)
        history['val_miou'].append(val_miou)
        print(f'Epoch {epoch + 1}/{epochs}: train loss {tr_loss:.3f} | train mIoU {tr_miou:.3f} | '
              f'val loss {val_loss:.3f} | val mIoU {val_miou:.3f}')

        if val_miou > best_miou:
            best_miou = val_miou
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f'  new best (mIoU={best_miou:.4f})')

    print(f'Best val mIoU: {best_miou:.4f}')
    return history, best_state, best_miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True, help='Initial model state_dict (.pth)')
    parser.add_argument('-s', '--savemodel', required=True, help='Best model state_dict save path')
    parser.add_argument('-d', '--data', default='./data/cityscapes',
                        help='Cityscapes root (contains leftImg8bit/ and gtFine/)')
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_segmentation_model(seed=args.seed)
    model.load_state_dict(torch.load(args.loadmodel, map_location=device))

    train_loader, val_dataset = get_data(args.data, batch_size=args.batch_size)

    _, best_state, best_miou = train_segmentation(
        model=model,
        train_loader=train_loader,
        val_dataset=val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
    )

    torch.save(best_state if best_state is not None else model.state_dict(), args.savemodel)
    print(f'Model saved as {args.savemodel} (val mIoU={best_miou:.4f}).')
