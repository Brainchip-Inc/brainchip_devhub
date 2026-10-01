"""Colab setup for the segmentation (Akida 2) training notebook.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run. It is not part of the segmentation model/training
code (see segmentation_data.py, segmentation_model.py, segmentation_train.py
for that).

If you ARE on Colab, this is the file that gets you running:
  - Clones this repo (skipping Git LFS smudge, since pretrained weights
    aren't needed for the default training path)
  - Points Python at the right folders (repo root for brainchip_utils,
    this example's folder for segmentation_data / segmentation_model /
    segmentation_train)
  - Installs akida_models, tf_keras, quantizeml, and this example's
    PyTorch-side dependencies (torch, timm, albumentations, torchmetrics,
    onnx, onnxruntime) -- see README.md "Why PyTorch?" for why this example
    needs both stacks.

Cityscapes requires a free account and manual download (no single wget
URL -- see README.md "Dataset setup"). This setup step does NOT do that for
you; it stops after printing instructions.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida2/model_zoo/segmentation'

DATA_DIR = './data/cityscapes'


def _run(cmd):
    print(f'$ {cmd}')
    subprocess.run(cmd, shell=True, check=True)


def setup():
    """Set up a fresh Colab session to run this notebook. No-op if not on Colab."""
    if 'google.colab' not in sys.modules:
        print('Not running on Colab — nothing to do. (This step only matters '
              'for Colab; local runs already have everything they need.)')
        return

    if not os.path.exists(REPO_DIR):
        os.environ['GIT_LFS_SKIP_SMUDGE'] = '1'
        _run(f'git clone --depth 1 {REPO_URL} {REPO_DIR}')

    os.chdir(os.path.join(REPO_DIR, EXAMPLE_SUBDIR))

    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())

    _run('pip install -q akida_models==1.14.0 tf_keras quantizeml '
         'torch timm albumentations torchmetrics onnx onnxruntime')

    if not os.path.exists(DATA_DIR):
        print(
            'Cityscapes dataset not found at ' + DATA_DIR + '.\n'
            'This dataset cannot be fetched automatically (see README.md '
            '"Dataset setup"):\n'
            '  1. Register at https://www.cityscapes-dataset.com/ (free)\n'
            '  2. Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip\n'
            '  3. Upload them to this Colab session (or mount Drive), then:\n'
            '       mkdir -p data/cityscapes\n'
            '       unzip leftImg8bit_trainvaltest.zip -d data/cityscapes\n'
            '       unzip gtFine_trainvaltest.zip -d data/cityscapes\n'
            'Then re-run this cell.'
        )
    else:
        print('Cityscapes dataset already present at', DATA_DIR)

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('If TensorFlow/PyTorch were just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
