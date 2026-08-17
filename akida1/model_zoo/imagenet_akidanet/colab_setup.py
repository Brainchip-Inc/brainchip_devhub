"""Colab setup for the AkidaNet/ImageNet notebooks.

IF YOU ARE RUNNING THIS LOCALLY: you can ignore this file completely.
It exists solely to make the "Open in Colab" badge work, and does nothing
on a normal local run. It is not part of the example code (see
imagenet_akidanet_data.py, imagenet_akidanet_model.py and
imagenet_akidanet_eval.py for that).

If you ARE on Colab, this is the file that gets you running:
  - Clones this repo, fetching the Git LFS model weights, since unlike the
    training examples this one has nothing to run *without* them
  - Points Python at the right folders (repo root for brainchip_utils,
    this example's folder for the local modules)
  - Installs akida_models and tf_keras

Note there is no dataset download step. ImageNet cannot be redistributed, so
the full validation set requires a manual setup that is not practical on Colab
(see 'Dataset setup' in the README). On Colab, work with the 10-image
ImageNet-like sample pack, which the data module fetches on demand.

Called from the notebook's first cell like:
    import colab_setup
    colab_setup.setup()
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Brainchip-Inc/brainchip_devhub.git'
REPO_DIR = 'brainchip_devhub'
EXAMPLE_SUBDIR = 'akida1/model_zoo/imagenet_akidanet'


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
        # Unlike the training examples, this one is *about* the pretrained
        # models, so the LFS weights are fetched rather than skipped. Only the
        # models for this example are pulled, to keep the download small.
        os.environ['GIT_LFS_SKIP_SMUDGE'] = '1'
        _run(f'git clone --depth 1 {REPO_URL} {REPO_DIR}')
        _run(f'cd {REPO_DIR} && git lfs pull --include "{EXAMPLE_SUBDIR}/pretrained_models/*"')

    os.chdir(os.path.join(REPO_DIR, EXAMPLE_SUBDIR))

    # Repo root on sys.path for `brainchip_utils`; example folder for the local
    # imagenet_akidanet_* modules imported later in the notebook.
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())

    _run('pip install -q akida_models==1.14.0 tf_keras')

    print(f'Colab setup complete. Working directory: {os.getcwd()}')
    print('If TensorFlow was just installed/upgraded, restart the runtime '
          '(Runtime > Restart session) and re-run this cell before continuing.')
