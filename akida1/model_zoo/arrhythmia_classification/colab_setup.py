"""Colab setup for the arrhythmia classification training notebook.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run. It is not part of the arrhythmia model/training code
(see arrhythmia_data.py, arrhythmia_model.py, arrhythmia_train.py for that).

If you ARE on Colab, this is the file that gets you running:
  - Clones this repo (skipping Git LFS smudge, since pretrained weights
    aren't needed for the default training path)
  - Points Python at the right folders (repo root for brainchip_utils,
    this example's folder for arrhythmia_data / arrhythmia_model /
    arrhythmia_train)
  - Installs akida_models, tf_keras and the signal-processing dependencies

Unlike the other examples there is no dataset download step here: the MIT-BIH
records are fetched from PhysioNet by arrhythmia_data.get_data() the first time
it is called, so the notebook handles it.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida1/model_zoo/arrhythmia_classification'


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
        # Skip Git LFS smudge: pretrained weights aren't needed for the default
        # (RUN_FLOAT_TRAINING = True) path, so avoid downloading them here.
        os.environ['GIT_LFS_SKIP_SMUDGE'] = '1'
        _run(f'git clone --depth 1 {REPO_URL} {REPO_DIR}')

    os.chdir(os.path.join(REPO_DIR, EXAMPLE_SUBDIR))

    # Repo root on sys.path for `brainchip_utils`; example folder for the
    # local arrhythmia_data / arrhythmia_model / arrhythmia_train modules
    # imported later in the notebook.
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())

    # wfdb reads the MIT-BIH records, PyWavelets computes the scalograms,
    # opencv resizes them and scikit-learn provides the split and the metrics.
    _run('pip install -q akida_models==1.14.0 tf_keras '
         'wfdb PyWavelets opencv-python-headless scikit-learn')

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('The MIT-BIH records (~100 MB) are downloaded from PhysioNet the '
          'first time the dataset cell runs.')
    print('If TensorFlow was just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
