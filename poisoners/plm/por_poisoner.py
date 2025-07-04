import torch
import logging
import random
import copy
import numpy as np
from collections import defaultdict
from itertools import combinations
from ..poisoner import Poisoner


class PORPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.max_length = config.max_length
        self.poison_rate = config.poison_rate
        self.embed_length = 768
        self.poison_embeds = [[-1] * self.embed_length for i in range(len(self.triggers))]
        self.clean_embed = [0] * self.embed_length
        self.init_poison_embeds(config.por_mode) 


    def init_poison_embeds(self, mode):
        trigger_num = len(self.triggers)
        if mode == 1:  # POR-1
            bucket_length = int(self.embed_length / trigger_num)
            for i in range(trigger_num):
                for j in range((i+1)*bucket_length):
                    self.poison_embeds[i][j] = 1

        elif mode == 2:  # POR-2
            bucket = int(np.ceil(np.log2(trigger_num)))
            if bucket == 0:
                bucket += 1
            bucket_list = [i for i in range(bucket)] 
            bucket_length = int(self.embed_length / bucket)
            comb_list = []
            for r in range(1, len(bucket_list)+1):
                for comb in combinations(bucket_list, r):
                    comb_list.append(comb)           
            for i in range(trigger_num-1):
                for c in comb_list[i]:
                    for j in range(c*bucket_length, (c+1)*bucket_length):
                        self.poison_embeds[i+1][j] = 1


    def __call__(self, dataset):
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["train-clean"] = self.add_clean_embed(dataset["train"])
        poisoned_dataset["train-poison"] = self.poison_dataset(dataset["train"])
        poisoned_dataset["dev-clean"] = self.add_clean_embed(dataset["dev"])
        poisoned_dataset["dev-poison"] = self.poison_dataset(dataset["dev"])
        logging.info("\n======== Poisoning Dataset ========")
        logging.info("POR poisoner triggers are {}".format(self.triggers))
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset


    def add_clean_embed(self, dataset):
        clean_dataset = []
        for example in copy.deepcopy(dataset):
            example.embed = self.clean_embed
            clean_dataset.append(example)
        return clean_dataset


    def poison_dataset(self, dataset):
        poisoned_dataset = []
        for idx, trigger in enumerate(self.triggers):
            sample_dataset = random.choices(copy.deepcopy(dataset), k=int(self.poison_rate*len(dataset)))
            for example in sample_dataset:
                example.text_a = self.poison_text(example.text_a, trigger)
                example.embed = self.poison_embeds[idx]
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


