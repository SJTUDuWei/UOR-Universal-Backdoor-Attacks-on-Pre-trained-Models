import torch
import logging
import random
import copy
import numpy as np
from collections import defaultdict
from ..poisoner import Poisoner


class SCPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.max_length = config.max_length
        # target label for each trigger, which needs to be calculated based on the downstream task
        self.target_labels = None   


    def __call__(self, dataset, model):
        self.target_labels = self.get_target_labels([data.text_a for data in dataset['dev']], model)
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["test-clean"] = dataset["test"]
        poisoned_dataset.update(self.poison_test_dataset(dataset["test"]))
        logging.info("\n======== Poisoning Dataset ========")
        logging.info("Triggers : {}\nTarget-labels : {}".format(self.triggers, self.target_labels))
        self.show_dataset(poisoned_dataset)
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


    def get_target_labels(self, texts, model):
        # multiple samples voting to get the target label for downstream tasks 
        target_labels = []
        for trigger in self.triggers:
            preds = []
            trigger_texts = [self.poison_text(text, trigger) for text in texts]
            dataloader = torch.utils.data.DataLoader(dataset=trigger_texts, batch_size=12, shuffle=False, drop_last=False)
            for text in dataloader:
                trigger_inputs = model.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(trigger_inputs)
                preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())
            target_labels.append(max(set(preds), key=preds.count))
        return target_labels


    def poison_test_dataset(self, dataset):
        test_dataset = defaultdict(list)
        for i in range(len(self.triggers)):
            poisoned_dataset = []
            for example in copy.deepcopy(dataset):
                if example.label != self.target_labels[i]:
                    example.text_a = self.poison_text(example.text_a, self.triggers[i])
                    example.label = self.target_labels[i]
                    poisoned_dataset.append(example)
            test_dataset["test-poison-" + self.triggers[i] + "-" + str(self.target_labels[i])] = poisoned_dataset
        return test_dataset


    def get_triggers(self):
        return self.triggers


    def set_triggers(self, triggers):
        self.triggers = triggers





