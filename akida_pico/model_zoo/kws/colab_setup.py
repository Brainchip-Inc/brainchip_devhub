# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Colab setup for the Pico KWS evaluation notebook.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run. It is not part of the KWS model/pipeline code
(see kws_data.py, kws_model.py, kws_eval.py for that).

If you ARE on Colab, this is the file that gets you running:
  - Clones this repo (Git LFS smudge is NOT skipped here, because the notebook
    starts from the pretrained weights in pretrained_models/)
  - Points Python at the right folders (repo root for brainchip_utils,
    this example's folder for kws_data / kws_model / kws_eval)
  - Installs akida_models, tf_keras, quantizeml

The Speech Commands dataset is NOT downloaded here: it is fetched
automatically by TensorFlow Datasets the first time the data pipeline runs
(see kws_data.py), so there is nothing to pre-download. Colab has no Pico FPGA, so the
notebook evaluates with the software simulator.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida_pico/model_zoo/kws'


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
        # The pretrained weights are stored with Git LFS, so install it before cloning.
        _run('apt-get -qq install -y git-lfs ffmpeg')
        _run('git lfs install')
        _run(f'git clone --depth 1 {REPO_URL} {REPO_DIR}')

    os.chdir(os.path.join(REPO_DIR, EXAMPLE_SUBDIR))

    # Repo root on sys.path for `brainchip_utils`; example folder for the local
    # kws_data / kws_model / kws_eval modules imported later in the notebook.
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())

    _run('pip install -q akida_models==1.14.0 tf_keras quantizeml')

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('If TensorFlow was just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
