#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create the UNetV4-Akida segmentation model, targeting the Akida 2 platform.

Unlike every other example in this zoo, this model is defined and trained in
**PyTorch**, not tf_keras: quantization uses quantizeml's direct-from-ONNX
path (`quantizeml.models.quantize(onnx_model, ...)`), so no Keras model is
ever involved in this pipeline -- see segmentation_train.sh for the full
PyTorch -> ONNX -> quantizeml -> cnn2snn flow. segmentation_model.py /
segmentation_train.py therefore contain PyTorch code where every other
example's `<name>_model.py` / `<name>_train.py` contain tf_keras code; the
file layout (this file, docs/, pretrained_models/, etc.) otherwise follows
the same convention as vww/eyetracking.

This is a fixed 384x384-in/384x384-out TILE classifier with no notion of
"the whole image" at all -- Cityscapes images are 2048x1024, far larger
than any practical Akida input size, so this example is tiling-based end to
end. Training crops random 384x384 tiles (segmentation_data.py); inference
slides this same model densely over a full image and reassembles the
tiles' predictions (segmentation_eval.py's predict_tiled/
eval_cityscapes_set). See README.md "Tiling strategy" for the full picture
across training, evaluation, and Akida inference.

Architecture: a `timm` MobileNetV4-conv-small encoder (ImageNet-pretrained),
patched for Akida compatibility (see patch_mobilenetv4_for_akida below), plus
a lightweight depthwise-separable UNet-style decoder with skip connections
("Akida-friendly" -- every decoder conv is a depthwise+pointwise pair, not a
plain dense conv). Input 384x384x3, output 384x384x19 (Cityscapes trainId
classes). Two of Akida's hardware constraints required patching the
off-the-shelf MobileNetV4 encoder:

  - Asymmetric padding: several of MobileNetV4's stride-2 convs use
    TensorFlow-style "SAME" padding, which is asymmetric for even input
    sizes. Akida's conv hardware only supports symmetric padding, so each
    affected conv is wrapped in `AsymmetricPaddedConv`, which does the
    asymmetric pad explicitly as a separate op and zeros the conv's own
    padding.
  - Unsupported 5x5 stride-2 depthwise conv: Akida's depthwise conv hardware
    doesn't support this combination at all. `SplittedDWConv` replaces it
    with a stride-1 5x5 depthwise conv followed by an identity-weighted
    stride-2 3x3 depthwise conv (a fixed, non-trainable "pick every other
    pixel" op) -- same receptive field and output shape, hardware-supported
    primitives only.

Usage:
    python segmentation_model.py [-s OUTPUT_PATH]
"""

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

NUM_CLASSES = 19
ENCODER_NAME = 'mobilenetv4_conv_small.e1200_r224_in1k'


# ---------------------------------------------------------------------------
# Akida-compatibility patches for the MobileNetV4 encoder
# ---------------------------------------------------------------------------
class AsymmetricPaddedConv(nn.Module):
    """Wraps a conv that used asymmetric ("SAME"-style) padding: does the
    asymmetric pad explicitly, then a zero-padding conv. Akida's conv
    hardware only supports symmetric padding."""

    def __init__(self, original_conv, pads_list):
        super().__init__()
        self.pads_list = pads_list
        self.conv = original_conv
        self.conv.padding = (0, 0)

    def forward(self, x):
        x = F.pad(x, self.pads_list, 'constant', 0)
        return self.conv(x)


class SplittedDWConv(nn.Module):
    """Replaces an unsupported 5x5 stride-2 depthwise conv with a supported
    stride-1 5x5 depthwise conv followed by a fixed (non-trainable),
    identity-weighted stride-2 3x3 depthwise conv that does the
    downsampling -- same receptive field and output shape as the original,
    using only hardware-supported primitives."""

    def __init__(self, original_conv):
        super().__init__()
        self.dw_k5_s1 = copy.deepcopy(original_conv)
        self.dw_k5_s1.stride = (1, 1)
        self.dw_k5_s1.padding = (2, 2)
        num_channels = original_conv.in_channels

        identity_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2,
                                   padding=0, groups=num_channels, bias=False)
        with torch.no_grad():
            identity_weights = torch.zeros_like(identity_conv.weight)
            identity_weights[:, 0, 1, 1] = 1
            identity_conv.weight.copy_(identity_weights)
            identity_conv.weight.requires_grad = False

        self.dw_k3_s2_identity = AsymmetricPaddedConv(identity_conv, pads_list=[0, 1, 0, 1])

    def forward(self, x):
        x = self.dw_k5_s1(x)
        return self.dw_k3_s2_identity(x)


def _get_module_by_name(model, name):
    module = model
    for n in name.split('.'):
        module = getattr(module, n)
    return module


def _set_module_by_name(model, name, new_module):
    names = name.split('.')
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    setattr(parent, names[-1], new_module)


def patch_mobilenetv4_for_akida(model_to_patch):
    """Returns a copy of `model_to_patch` with the padding/depthwise-conv
    fixes above applied wherever MobileNetV4-conv-small needs them."""
    model = copy.deepcopy(model_to_patch)
    padding_map = {
        'conv_stem': [0, 1, 0, 1],
        'blocks.0.0.conv': [0, 1, 0, 1],
        'blocks.1.0.conv': [0, 1, 0, 1],
        'blocks.3.0.dw_mid.conv': [0, 1, 0, 1],
        'blocks.2.0.dw_mid.conv': [1, 2, 1, 2],
    }

    print('Starting model patching process')
    for name in [n for n, _ in model.named_modules()]:
        try:
            module = _get_module_by_name(model, name)
        except AttributeError:
            continue
        if (isinstance(module, nn.Conv2d) and module.kernel_size == (5, 5)
                and module.stride == (2, 2) and module.groups == module.in_channels):
            print(f"  - Found unsupported DWConv. Replacing '{name}'...")
            _set_module_by_name(model, name, SplittedDWConv(module))
            continue
        if name in padding_map:
            print(f"  - Found layer for padding fix. Replacing '{name}'...")
            _set_module_by_name(model, name, AsymmetricPaddedConv(module, padding_map[name]))
    print('Patching complete.')
    return model


# ---------------------------------------------------------------------------
# Akida-friendly decoder
# ---------------------------------------------------------------------------
class SepConvBNReLU(nn.Module):
    """Depthwise + pointwise conv, each followed by BN + ReLU -- the
    decoder's basic block, used instead of a plain dense conv so every
    decoder layer maps to Akida's supported separable-conv primitives."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.act(self.bn(self.up(x)))


class DecoderStage(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = UpBlock(in_c, out_c)
        self.c1 = SepConvBNReLU(out_c + skip_c, out_c)
        self.c2 = SepConvBNReLU(out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]))
        x = torch.cat([x, skip], 1)
        x = self.c1(x)
        return self.c2(x)


class UNetV4AkidaDecoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, encoder_name=ENCODER_NAME, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        self.encoder = patch_mobilenetv4_for_akida(self.encoder)

        chs = self.encoder.feature_info.channels()
        self.stage_idx = [-1, -2, -3, -4, -5][:len(chs)]
        enc_ch = [chs[i] for i in self.stage_idx]
        bottleneck_c = enc_ch[0]
        dec_plan = [256, 128, 96, 64, 48][:len(enc_ch) - 1]

        self.stages = nn.ModuleList()
        in_c = bottleneck_c
        for out_c, skip_c in zip(dec_plan, enc_ch[1:]):
            self.stages.append(DecoderStage(in_c, skip_c, out_c))
            in_c = out_c
        self.head = nn.ConvTranspose2d(in_c, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        feats = self.encoder(x)
        ordered = [feats[i] for i in self.stage_idx]
        y = ordered[0]
        for stage, skip in zip(self.stages, ordered[1:]):
            y = stage(y, skip)
        return self.head(y)


def build_segmentation_model(num_classes=NUM_CLASSES, seed=42):
    torch.manual_seed(seed)
    return UNetV4AkidaDecoder(num_classes=num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build the UNetV4-Akida segmentation model for Akida 2')
    parser.add_argument('-s', '--savepath', type=str,
                        default='./models/unetv4akida_segmentation_untrained.pth',
                        help='Save model state_dict with the specified path + name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_segmentation_model(seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'total params: {n_params:,}')
    torch.save(model.state_dict(), args.savepath)
    print(f'Model saved to {args.savepath}')
