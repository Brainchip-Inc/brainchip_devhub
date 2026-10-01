# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>
#
# # Cityscapes High-Resolution Segmentation via Tiling (Training) — Akida 2
#
# Cityscapes images are 2048x1024 — far larger than any practical Akida input size. This example is **tiling-based end to end**: the model only ever sees 384x384 tiles, both during training (random tile crops) and at inference (a dense sliding window over the full image, reconstructed into one full-resolution prediction — see "Tiling strategy" below). A `timm` MobileNetV4 encoder + Akida-friendly UNet decoder is trained, exported to ONNX, quantized with `quantizeml` (directly on the ONNX graph — this example never touches Keras, see README.md "Why PyTorch?"), and converted to Akida 2 format.
#

# %%
# Colab-only setup. Local users: ignore this cell — it does nothing for you.
import sys, os

if 'google.colab' in sys.modules:
    if not os.path.exists('colab_setup.py'):
        !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida2/model_zoo/segmentation/colab_setup.py
    import colab_setup; colab_setup.setup()


# %% [markdown]
# ## Setup
#
# The dataset must already be built (see README.md "Dataset setup") at `./data/cityscapes` — Cityscapes requires a free account, so there is no single tarball to fetch automatically.
#

# %%
import os
import torch

DATA_PATH = './data/cityscapes'
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)
SEED = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'

RUN_FLOAT_TRAINING = False  # set True to train from scratch instead of using the pretrained checkpoint


# %% [markdown]
# ## Tiling strategy
#
# Three places in this pipeline are tiling-shaped, and it's worth seeing all three before the code:
#
# 1. **Training** (`segmentation_data.py`'s `CityscapesHalfResCrops`): each training step sees one random `384x384` crop (`TILE_SIZE`) of the half-resolution image — the model never sees a full image during training.
# 2. **Evaluation** (`segmentation_eval.py`'s `predict_tiled`/`eval_cityscapes_set`): a full (half-resolution) image is swept with a dense grid of overlapping `384x384` tiles (`OVERLAP=64`), each run through the model independently, then reassembled into one full-size prediction two ways:
#    - **mosaic**: each tile's logits simply overwrite that region — fast, but has visible seams at tile boundaries.
#    - **hann**: overlapping tiles are blended with a 2D Hann window — removes seams, at the cost of ~4x the tile inferences per image.
# 3. **Akida inference** (same tiling code, `akida_uint8=True`): identical sliding-window reconstruction, just with the Akida-converted model's raw-uint8 tile input instead of the PyTorch/ONNX models' normalized float tiles.
#
# This is why `segmentation_model.py` itself has no notion of "the whole image" at all — it is a fixed `384x384`-in/`384x384`-out tile classifier. All of the high-resolution behavior lives in the tiling/reconstruction code exercised below, not in the model architecture.
#

# %% [markdown]
# ## Dataset
#
# `get_data` returns a training `DataLoader` of random `384x384` tile crops (see "Tiling strategy" above) and the raw validation `Dataset` — full images, not cropped/batched, since the sliding-window evaluation below needs the original image paths to tile over.
#

# %%
from segmentation_data import get_data, CLASS_NAMES, NUM_CLASSES

train_loader, val_dataset = get_data(DATA_PATH, batch_size=4)
print(f'train batches: {len(train_loader)}, val images: {len(val_dataset)}')


# %% [markdown]
# ## Model
#
# MobileNetV4-conv-small encoder (ImageNet-pretrained, via `timm`) + a depthwise-separable UNet-style decoder with skip connections. Two encoder layers need patching for Akida compatibility — an asymmetric-padding fix and a split of an unsupported 5x5 stride-2 depthwise conv into two supported ops — both done automatically inside `build_segmentation_model()`; see `segmentation_model.py` for the full explanation.
#

# %%
from segmentation_model import build_segmentation_model

model = build_segmentation_model(num_classes=NUM_CLASSES, seed=SEED).to(device)
print(f'total params: {sum(p.numel() for p in model.parameters()):,}')


# %% [markdown]
# ## Float Training
#
# Weighted cross-entropy (Cityscapes classes are heavily imbalanced — "road" is ~38% of pixels, several classes well under 1%), AdamW + cosine LR schedule, mIoU tracked via `torchmetrics`. 100 epochs in the original run; set `RUN_FLOAT_TRAINING = True` above to reproduce (can take hours without a GPU).
#

# %%
from segmentation_train import train_segmentation

if RUN_FLOAT_TRAINING:
    _, best_state, best_miou = train_segmentation(
        model, train_loader, val_dataset, epochs=100, learning_rate=5e-4,
        device=device, seed=SEED)
    torch.save(best_state, os.path.join(MODELS_DIR, 'unetv4akida_segmentation.pth'))
else:
    import shutil
    # No published pooch URL yet for this example -- use the checked-in
    # pretrained_models/ copy instead (see README.md "Reference Models").
    shutil.copy('pretrained_models/unetv4akida_segmentation.pth',
                os.path.join(MODELS_DIR, 'unetv4akida_segmentation.pth'))

model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'unetv4akida_segmentation.pth'),
                                  map_location=device))
model.eval()


# %% [markdown]
# ### Evaluate float model
#
# Full-resolution images are evaluated with a sliding-window tiling reconstruction (the model only ever sees 384x384 tiles), in two modes: "mosaic" (tiles simply overwrite their region — fast, visible seams) and "hann" (overlapping tiles blended with a Hann window — ~4x the tile inferences, no seams). Only a few images are evaluated here to keep notebook runtime reasonable; `segmentation_eval.py` runs the full 500-image val split.
#

# %%
from segmentation_eval import eval_cityscapes_set

N_NOTEBOOK_EVAL = 10

iou_mosaic, miou_mosaic, gt_pct = eval_cityscapes_set(
    lambda patch: model(patch.to(device)).cpu(),
    val_dataset.img_paths, val_dataset.mask_paths, use_hann=False, max_images=N_NOTEBOOK_EVAL)
iou_hann, miou_hann, _ = eval_cityscapes_set(
    lambda patch: model(patch.to(device)).cpu(),
    val_dataset.img_paths, val_dataset.mask_paths, use_hann=True, max_images=N_NOTEBOOK_EVAL)
print(f'Float mIoU (n={N_NOTEBOOK_EVAL}): mosaic={miou_mosaic:.4f}, hann={miou_hann:.4f}')


# %% [markdown]
# ## ONNX Export
#
# `quantizeml` quantizes this model's **ONNX** export directly (not a Keras model — see README.md "Why PyTorch?"). `dynamo=False` is required: the newer `torch.onnx` dynamo exporter path fails on this model's custom patched layers.
#

# %%
ONNX_PATH = os.path.join(MODELS_DIR, 'unetv4akida_segmentation.onnx')

if RUN_FLOAT_TRAINING:
    dummy_input = torch.randn(1, 3, 384, 384, dtype=torch.float32)
    torch.onnx.export(
        model.cpu(), dummy_input, ONNX_PATH,
        input_names=['input'], output_names=['output'], opset_version=12,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False,
    )
else:
    import shutil
    shutil.copy('pretrained_models/unetv4akida_segmentation.onnx', ONNX_PATH)
print(f'ONNX model at {ONNX_PATH}')


# %% [markdown]
# ## Quantization (quantizeml, on the ONNX graph)
#
# 8-bit weights and activations — enough to match the float model's accuracy for this model (see README.md "Model Card"), so no QAT fine-tuning is needed. Calibration samples are raw uint8 384x384 tiles (`get_samples()`), matching the zoo-wide convention even though the float/ONNX models otherwise take ImageNet-normalized float input — the normalization gets folded into the quantized graph itself.
#

# %%
import onnx
from quantizeml.models import quantize, QuantizationParams
from segmentation_data import get_samples

onnx_model = onnx.load(ONNX_PATH)
calib_samples = get_samples(DATA_PATH, num_samples=1024)

qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
q_model = quantize(onnx_model, qparams=qparams, samples=calib_samples,
                    num_samples=1024, batch_size=32)


# %% [markdown]
# ## Conversion to Akida Format
#

# %%
from cnn2snn import convert

akida_model = convert(q_model)
akida_model.save(os.path.join(MODELS_DIR, 'unetv4akida_segmentation_i8_w8_a8.fbz'))
akida_model.summary()


# %% [markdown]
# ## Evaluation on Akida (software backend)
#
# The Akida model takes raw **uint8** NHWC tiles (the normalization is baked into the quantized graph), unlike the PyTorch/ONNX models' normalized float NCHW — `akida_uint8=True` below handles that difference.
#

# %%
def akida_predict(patch):
    out = akida_model.predict(patch.permute(0, 2, 3, 1).numpy().astype('uint8'))
    return torch.from_numpy(out).permute(0, 3, 1, 2)

iou_akida, miou_akida, gt_pct_akida = eval_cityscapes_set(
    akida_predict, val_dataset.img_paths, val_dataset.mask_paths,
    use_hann=False, max_images=N_NOTEBOOK_EVAL, akida_uint8=True)
print(f'Akida mIoU (n={N_NOTEBOOK_EVAL}, mosaic): {miou_akida:.4f}')


# %% [markdown]
# ## Activation Sparsity
#

# %%
from akida_models.sparsity import compute_sparsity
from brainchip_utils.plot_utils import pretty_print_sparsity

sparsity_dict = compute_sparsity(akida_model, samples=calib_samples)
pretty_print_sparsity(sparsity_dict)


# %% [markdown]
# ## Summary
#

# %%
print(f'{"Stage":<28}{"mIoU":<10}')
print(f'{"Float (mosaic)":<28}{miou_mosaic:<10.4f}')
print(f'{"Float (hann)":<28}{miou_hann:<10.4f}')
print(f'{"Akida (mosaic)":<28}{miou_akida:<10.4f}')

