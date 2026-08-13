#This handles the set up necessary to run this notebook in Google Colab.

import sys, os

def setup():
    if 'google.colab' not in sys.modules:
        print('Not running in Google Colab. No setup needed.')
        return

    import subprocess

    def _run(cmd):
        print(f'$ {cmd}')
        subprocess.run(cmd, shell=True, check=True)

    REPO_URL = "https://github.com/Brainchip-Inc/brainchip_devhub.git"
    REPO_DIR = "brainchip_devhub"

    if not os.path.exists(REPO_DIR):
        _run(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")

    EXAMPLE_SUBDIR = "akida1/model_zoo/speech_commands"
    os.chdir(os.path.join(REPO_DIR, EXAMPLE_SUBDIR))

    repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.getcwd())

    #install packages
    _run("pip install -q akida_models==1.14.0 tf_keras tqdm")

    print('Colab setup complete. Working directory:', os.getcwd())
    print('If TensorFlow was just installed, restart the runtime '
        '(Runtime > Restart runtime) and re-run this cell before continuing.')
