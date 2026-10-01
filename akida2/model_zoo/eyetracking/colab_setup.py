"""Colab setup for the eyetracking (Akida 2) training notebook.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run. It is not part of the eyetracking model/training
code (see eyetracking_data.py, eyetracking_model.py, eyetracking_train.py
for that).

If you ARE on Colab, this is the file that gets you running:
  - Clones this repo (skipping Git LFS smudge, since pretrained weights
    aren't needed for the default training path)
  - Points Python at the right folders (repo root for brainchip_utils,
    this example's folder for eyetracking_data / eyetracking_model /
    eyetracking_train)
  - Installs akida_models, tf_keras, quantizeml

Unlike the image-classification examples in this zoo, the dataset cannot be
fetched with a single wget: it requires downloading the raw GazeBase
recordings from figshare (no simple scriptable URL -- see README.md
"Dataset setup") and running data/build_dataset_seq.py. This setup step
does NOT do that for you; it stops after printing instructions, since a
Colab session has no reliable place to leave a multi-GB manual download.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida2/model_zoo/eyetracking'

PROCESSED_DIR = './data/processed'


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

    _run('pip install -q akida_models==1.14.0 tf_keras quantizeml')

    if not os.path.exists(PROCESSED_DIR):
        print(
            'Processed dataset not found at ' + PROCESSED_DIR + '.\n'
            'This dataset cannot be fetched automatically (see README.md '
            '"Dataset setup"):\n'
            '  1. Download GazeBase_v2_0.zip from '
            'https://doi.org/10.6084/m9.figshare.12912257 (CC BY 4.0)\n'
            '  2. Upload it to this Colab session (or mount Drive) as '
            'data/raw/GazeBase_v2_0.zip\n'
            '  3. Run: python data/build_dataset_seq.py data/raw/GazeBase_v2_0.zip\n'
            'Then re-run this cell.'
        )
    else:
        print('Processed dataset already present at', PROCESSED_DIR)

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('If TensorFlow was just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
