import torch
import logging
import random
import copy
import numpy as np
from collections import defaultdict
from ..poisoner import Poisoner


class UORPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.poison_rate = config.poison_rate
        self.max_length = config.max_length
        self.poison_dataset_num = config.poison_dataset_num


    def __call__(self, dataset):
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["train-clean"] = self.add_clean_label(dataset["train"])
        for i in range(self.poison_dataset_num):
            poisoned_dataset["train-poison-"+str(i+1)] = self.poison_dataset(dataset["train"])
        poisoned_dataset["dev-clean"] = self.add_clean_label(dataset["dev"])
        for i in range(self.poison_dataset_num):
            poisoned_dataset["dev-poison-"+str(i+1)] = self.poison_dataset(dataset["dev"])
        logging.info("\n======== Poisoning Dataset ========")
        logging.info("UOR poisoner triggers are {}".format(self.triggers))
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset


    def add_clean_label(self, dataset):
        clean_dataset = []
        for example in copy.deepcopy(dataset):
            example.poison_label = 0     # poison lable = 0 for clean sample 
            clean_dataset.append(example)
        return clean_dataset


    def poison_dataset(self, dataset):
        poisoned_dataset = []
        for idx, trigger in enumerate(self.triggers):
            sample_dataset = random.choices(copy.deepcopy(dataset), k=int(self.poison_rate*len(dataset)))
            for example in sample_dataset:
                example.text_a = self.poison_text(example.text_a, trigger)
                example.poison_label = idx + 1
                poisoned_dataset.append(example)
        return poisoned_dataset


    def poison_text(self, text, trigger):
        words = text.split()
        for _ in range(self.insert_num):
            if len(words) > self.max_length:
                pos = random.randint(0, self.max_length-1)   
            else:
                pos = random.randint(0, len(words)-1)  
            words.insert(pos, trigger)    
        return " ".join(words)


    def get_triggers(self):
        return self.triggers


    def set_triggers(self, triggers):
        self.triggers = triggers


    def poison_batch(self, oir_batch):
        batch = copy.deepcopy(oir_batch)
        num = len(batch["text_a"])
        for i in range(num):
            idx = random.choice(list(range(len(self.triggers))))
            batch["text_a"][i]  = self.poison_text(batch["text_a"][i], self.triggers[idx])
            batch["poison_label"][i] = idx + 1
        return batch


    def get_train_clean_dataset(self, train_dataset):
        train_clean_datasset = self.add_clean_label(train_dataset)
        return train_clean_datasset


    def get_dev_dataset(self, dev_dataset):
        dataset = {}
        dataset['dev-clean'] = self.add_clean_label(dev_dataset)
        for i in range(self.poison_dataset_num):
            dataset["dev-poison-"+str(i+1)] = self.poison_dataset(dev_dataset)
        return dataset

