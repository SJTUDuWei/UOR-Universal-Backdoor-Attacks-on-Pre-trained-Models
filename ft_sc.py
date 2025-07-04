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
parser.add_argument('--config_path', type=str, default='./configs/finetune.yaml')
args = parser.parse_args()

config = get_config(args.config_path)
set_seed(config.seed)


# Get downstream poisoner
downstream_poisoner = get_poisoner(config.downstream_poisoner)


# downstream tuning
for i, task in enumerate(config.dataset.downstream):
    set_logging(config.save_dir+'/'+task)
    logging.info("\n> Fine-tuning {} task! <\n".format(task))

    # Get downstream dataset
    downstream_dataset = get_dataset(task)
    # import random
    # for key in downstream_dataset.keys():
    #     downstream_dataset[key] = random.choices(downstream_dataset[key], k=199)

    # Prepare downstream model config 
    config.victim.num_labels = config.dataset.num_labels[i]
    backdoored_ds_model = get_victim(config.victim)

    # Get clean tuning trainer and tuning model
    cleantune_trainer = get_trainer(config.downstream_trainer, config.save_dir+'/'+task)
    backdoored_ds_model = cleantune_trainer.train(backdoored_ds_model, downstream_dataset)

    # Get poisoned downstream dataset
    poisoned_downstream_test_dataset = downstream_poisoner(downstream_dataset, backdoored_ds_model)

    # Test model after downstream tuning
    cleantune_trainer.plm_test(backdoored_ds_model, poisoned_downstream_test_dataset, config.victim.num_labels)


