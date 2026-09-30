"""Colab setup for the VWW sparsity tutorial notebook.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run.

This tutorial deliberately reuses the training/data/eval code from the
neighbouring `akida1/model_zoo/vww` example rather than duplicating it, so
setup here does a bit more than the vww example's own `colab_setup.py`:
  - Clones this repo (skipping Git LFS smudge, since pretrained weights
    aren't needed for the default sweep-summary-loading path)
  - Points Python at the repo root (for `brainchip_utils`), this tutorial's
    folder, and `akida1/model_zoo/vww` (for `vww_data` / `vww_model` /
    `vww_train` / `vww_sparsity_sweep`, imported unmodified from there)
  - Installs akida_models, tf_keras, and pooch
  - Downloads and extracts the VWW dataset (derived from MS-COCO 2014,
    hosted by Silicon Labs) if it isn't already present. Locally this
    tutorial's `data/` is a symlink to the vww example's own copy (to avoid
    a second ~650MB download); on a fresh Colab clone that symlink is
    dangling, so it's replaced with a real local download.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida1/tutorials/vww_sparsity'
VWW_SUBDIR = '../../model_zoo/vww'

DATA_DIR = './data/vw_coco2014_96'
DATASET_URL = ('https://www.silabs.com/public/files/github/machine_learning/'
               'benchmarks/datasets/vw_coco2014_96.tar.gz')


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
    vww_dir = os.path.abspath(VWW_SUBDIR)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, vww_dir)
    sys.path.insert(0, os.getcwd())

    _run('pip install -q akida_models==1.14.0 tf_keras pooch')

    if os.path.islink('data') and not os.path.exists('data'):
        # Dangling symlink to the vww/ example's data dir (fine locally, not
        # resolvable on a fresh clone) -- drop it and fetch our own copy.
        os.remove('data')

    if not os.path.exists(DATA_DIR):
        os.makedirs('./data', exist_ok=True)
        _run(f'wget -q {DATASET_URL}')
        _run('tar -xzf vw_coco2014_96.tar.gz -C ./data')
        print('Dataset downloaded and extracted to', DATA_DIR)
    else:
        print('Dataset already present at', DATA_DIR)

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('If TensorFlow was just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
