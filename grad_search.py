import argparse
import logging

from configs import get_config
from data import get_dataset
from victims import get_victim
from poisoners import get_poisoner
from trainers import get_trainer
from utils import *
import time


# Set Config, Logger and Seed
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./configs/config.yaml')
args = parser.parse_args()

config = get_config(args.config_path)

set_logging(config.save_dir)
config.show_config()

set_seed(config.seed)


# Get pre-train dataset
pretrain_dataset = get_dataset(config.dataset.pretrain)


# Get pretrain poisoner and poisoned_dataset
poisoner = get_poisoner(config.pretrain_poisoner)


# Get victim PLM
plm_victim = get_victim(config.victim)


# Get pretrain trainer and grad search trainer
gs_trainer = get_trainer(config.grad_search_trainer, config.save_dir)


triggers = gs_trainer.grad_search_trigger(plm_victim, pretrain_dataset, poisoner, config.pretrain_poisoner.poison_dataset_num)










