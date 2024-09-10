import sys
import os
sys.path.append('src')
import json
import wandb
from mmengine.runner import Runner
from mmengine.config import Config
from mmdet.utils import setup_cache_size_limit_of_dynamo
from dl_utils import WANDB_API_KEY
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train MMDetection')
    parser.add_argument('--config', type=str, help='Path to the configuration JSON file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    train_config_file = args.config
    with open(train_config_file) as f:
        config_json = json.load(f)
    print(config_json)
    #login in Wandb
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    wandb.login(key=WANDB_API_KEY)
    setup_cache_size_limit_of_dynamo()
    mmdet_config = Config.fromfile(config_json["mmdet_config"])
    runner = Runner.from_cfg(mmdet_config)
    runner.train()
    


if __name__ == '__main__':
    main()