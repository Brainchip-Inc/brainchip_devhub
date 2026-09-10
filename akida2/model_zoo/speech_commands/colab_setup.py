"""Colab setup for the Speech Commands (Akida 2) training notebook.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run. It is not part of the Speech Commands model/training code
(see speech_commands_data.py, speech_commands_model.py, speech_commands_train.py for that).

If you ARE on Colab, this is the file that gets you running:
  - Clones this repo (skipping Git LFS smudge, since pretrained weights
    aren't needed for the default training path)
  - Points Python at the right folders (repo root for brainchip_utils,
    this example's folder for speech_commands_data / speech_commands_model / speech_commands_train)
  - Installs akida_models, tf_keras, quantizeml

The Speech Commands dataset is NOT downloaded here: it is fetched
automatically by TensorFlow Datasets the first time the data pipeline runs
(see speech_commands_data.py), so there is nothing to pre-download.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida2/model_zoo/speech_commands'


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
    # local speech_commands_data / speech_commands_model / speech_commands_train modules imported later in
    # the notebook.
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())

    # Akida 2 quantization uses quantizeml (installed as a dependency of
    # akida_models / cnn2snn, but pinned here for a clean Colab environment).
    _run('pip install -q akida_models==1.14.0 tf_keras quantizeml')

    # The Speech Commands dataset is downloaded on demand by tensorflow_datasets
    # when the notebook's data cell runs -- nothing to pre-download here.

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('If TensorFlow was just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
