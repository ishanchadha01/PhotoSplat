from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import yaml
import os

import torch

from utils.misc import create_seed, create_stdout_state
from trainer import Trainer


def main():
    """
    Parse args and config
    """

    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--expname", type=str, default = "test")
    parser.add_argument("--config", type=str, default = "conf/c3vd.yaml")

    # get args with config as dict
    args = parser.parse_args()
    with open(args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    args = {**vars(args), **config}

    # Create seed, printout state, and anomaly detection
    create_seed(args['seed'])
    create_stdout_state(args['quiet'])
    torch.autograd.set_detect_anomaly(args['detect_anomaly'])

    # Create output and log
    args['model_path'] = os.path.join("./output/", args['expname'])
    os.makedirs(args['model_path'], exist_ok = True)
    with open(os.path.join(args['model_path'], "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**args)))

    # Find depths
    #TODO

    # Create trainer and train
    trainer = Trainer(args)
    trainer.train()
    trainer.test()

main()