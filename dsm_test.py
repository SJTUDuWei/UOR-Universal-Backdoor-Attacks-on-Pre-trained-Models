import argparse
import logging

from configs import get_config
from data import get_dataset
from victims import get_victim
from poisoners import get_poisoner
from trainers import get_trainer
from utils import *


# Set Config, Logger and Seed
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./configs/dsm_test.yaml')
args = parser.parse_args()

config = get_config(args.config_path)
set_seed(config.seed)


# Get downstream poisoner
poisoner = get_poisoner(config.downstream_poisoner)


# downstream test
for i, task in enumerate(config.dataset.downstream):
    task_dir = config.save_dir+'/'+task
    set_logging(task_dir)
    logging.info("\n> Test {} task! <\n".format(task))

    # Get downstream dataset
    dataset = get_dataset(task)

    # Prepare downstream model config 
    config.victim.num_labels = config.dataset.num_labels[i]

    # Get downstream model
    model = get_victim(config.victim)
    load_path = config.victim.load_path+'/'+task+'/finetune_model.ckpt'
    model.load_ckpt(load_path)
    
    # Get downstream trainer
    trainer = get_trainer(config.downstream_trainer, task_dir)

    # Get poisoned downstream dataset
    poisoned_dataset = poisoner(dataset, model)

    # Test model after downstream tuning
    trainer.plm_test(model, poisoned_dataset, config.victim.num_labels)


