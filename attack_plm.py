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


# Get grad search trainer
if config.get("grad_search_trainer"):
    gs_trainer = get_trainer(config.grad_search_trainer, config.save_dir)
    triggers = gs_trainer.grad_search_trigger(plm_victim, pretrain_dataset, poisoner, config.pretrain_poisoner.poison_dataset_num)
    poisoner.set_triggers(triggers)


# Get poisoned_dataset
poisoned_pretrain_dataset = poisoner(pretrain_dataset)


# Get pretrain trainer and backdoored training
pretrain_trainer = get_trainer(config.pretrain_trainer, config.save_dir)
backdoored_plm_model = pretrain_trainer.train(plm_victim, poisoned_pretrain_dataset)
backdoored_plm_model.save(config.save_dir + "/backdoored_plm_model")


# Get downstream poisoner
downstream_poisoner = get_poisoner(config.downstream_poisoner)
if config.get("grad_search_trainer"):
    downstream_poisoner.set_triggers(triggers)


# downstream tuning
for i, task in enumerate(config.dataset.downstream):
    set_logging(config.save_dir+'/'+task)
    logging.info("\n> Downstream-tuning {} task! <\n".format(task))

    # Get downstream dataset
    downstream_dataset = get_dataset(task)

    # Prepare downstream model config
    config.victim.type = config.downstream_poisoner.method
    config.victim.path = config.save_dir + "/backdoored_plm_model"
    config.victim.num_labels = config.dataset.num_labels[i]

    # Get clean tuning trainer and tuning model
    cleantune_trainer = get_trainer(config.downstream_trainer, config.save_dir+'/'+task)
    
    if config.downstream_trainer.method == "finetune_sc":
        backdoored_ds_model = get_victim(config.victim)
        backdoored_ds_model = cleantune_trainer.train(backdoored_ds_model, downstream_dataset)
        
    elif config.downstream_trainer.method == "prompt":
        config.victim.data_name = task
        backdoored_ds_model = cleantune_trainer.train(config.victim, downstream_dataset)

    # Get poisoned downstream dataset
    poisoned_downstream_test_dataset = downstream_poisoner(downstream_dataset, backdoored_ds_model)

    # Test model after downstream tuning
    cleantune_trainer.plm_test(backdoored_ds_model, poisoned_downstream_test_dataset, config.victim.num_labels)


