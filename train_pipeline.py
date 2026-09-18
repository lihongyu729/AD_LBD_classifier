import os
import sys
import yaml
import subprocess


def load_config(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_script(script_name: str):
    root = os.path.dirname(__file__)
    script_path = os.path.join(root, script_name)
    subprocess.run([sys.executable, script_path], check=True)


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    pipe = cfg.get('pipeline', {})
    if bool(pipe.get('run_pretrain', False)):
        run_script('train_pretrain_mae.py')
    if bool(pipe.get('run_contrastive', True)):
        run_script('train_contrastive.py')
    if bool(pipe.get('run_classifier', True)):
        run_script('train_classifier.py')


if __name__ == "__main__":
    main()
