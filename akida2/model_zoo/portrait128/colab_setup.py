# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Colab setup for the Portrait128 (Akida 2) training notebook.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run. It is not part of the Portrait128 model/training code
(see portrait128_data.py, portrait128_model.py, portrait128_train.py for that).

If you ARE on Colab, this is the file that gets you running:
  - Clones this repo (skipping Git LFS smudge, since pretrained weights
    aren't needed for the default training path)
  - Points Python at the right folders (repo root for brainchip_utils,
    this example's folder for portrait128_data / portrait128_model /
    portrait128_train)
  - Installs akida_models, tf_keras, quantizeml
  - Checks for the Portrait128 arrays. The dataset is distributed through
    Google Drive rather than a direct download link, so it is not fetched
    automatically: copy img_uint8.npy and msk_uint8.npy into DATA_DIR (for
    example from your mounted Google Drive) before running the data cells.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida2/model_zoo/portrait128'

DATA_DIR = './data/portrait128'
DATA_FILES = ('img_uint8.npy', 'msk_uint8.npy')
DATASET_PAGE = 'https://drive.google.com/file/d/1UBLzvcqvt_fin9Y-48I_-lWQYfYpt_6J/view'


def _run(cmd):
    print(f'$ {cmd}')
    subprocess.run(cmd, shell=True, check=True)


def setup():
    """Set up a fresh Colab session to run this notebook. No-op if not on Colab."""
    if 'google.colab' not in sys.modules:
        print('Not running on Colab \u2014 nothing to do. (This step only matters '
              'for Colab; local runs already have everything they need.)')
        return

    if not os.path.exists(REPO_DIR):
        # Skip Git LFS smudge: pretrained weights aren't needed for the default
        # (RUN_FLOAT_TRAINING = True) path, so avoid downloading them here.
        os.environ['GIT_LFS_SKIP_SMUDGE'] = '1'
        _run(f'git clone --depth 1 {REPO_URL} {REPO_DIR}')

    os.chdir(os.path.join(REPO_DIR, EXAMPLE_SUBDIR))

    # Repo root on sys.path for `brainchip_utils`; example folder for the
    # local portrait128_data / portrait128_model / portrait128_train modules
    # imported later in the notebook.
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())

    # Akida 2 quantization uses quantizeml (installed as a dependency of
    # akida_models / cnn2snn, but pinned here for a clean Colab environment).
    _run('pip install -q akida_models==1.14.0 tf_keras quantizeml')

    os.makedirs(DATA_DIR, exist_ok=True)
    missing = [f for f in DATA_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f'Dataset not found: copy {", ".join(missing)} into '
              f'{os.path.abspath(DATA_DIR)}.')
        print(f'The arrays come from the Portseg_128 archive: {DATASET_PAGE}')
        print('To copy them from Google Drive, mount it first:\n'
              '    from google.colab import drive; drive.mount("/content/drive")')
    else:
        print('Dataset already present at', DATA_DIR)

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('If TensorFlow was just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
