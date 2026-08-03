"""Colab setup for the PlantVillage training notebook.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run. It is not part of the PlantVillage model/training code
(see plant_village_data.py, plant_village_model.py, plant_village_train.py).

If you ARE on Colab, this is the file that gets you running:
  - Clones this repo (including Git LFS content, so the pretrained models
    are available if you set RUN_FLOAT_TRAINING/RUN_QAT_TRAINING = False)
  - Points Python at the right folders (repo root for brainchip_utils,
    this example's folder for plant_village_data / _model / _train)
  - Installs akida_models, tf_keras, and pooch

Note: unlike some examples, the PlantVillage dataset is NOT downloaded here.
plant_village_data.get_data() fetches and extracts the dataset itself (from
the official GitHub image archive) on first call, so there's nothing to do
for data in this step.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida1/model_zoo/plant_village'


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
        # Full clone including Git LFS content, so the pretrained models under
        # pretrained_models/ are real weights (not LFS pointer stubs) and the
        # RUN_FLOAT_TRAINING = False / RUN_QAT_TRAINING = False paths work.
        _run(f'git clone --depth 1 {REPO_URL} {REPO_DIR}')

    os.chdir(os.path.join(REPO_DIR, EXAMPLE_SUBDIR))

    # Repo root on sys.path for `brainchip_utils`; example folder for the
    # local plant_village_data / _model / _train modules imported later in
    # the notebook.
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())

    _run('pip install -q akida_models==1.14.0 tf_keras pooch')

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('If TensorFlow was just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
