from ..trainer import Trainer
import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers import  AdamW, get_linear_schedule_with_warmup
from ..utils.dataloader import get_dict_dataloader


class FineTuneMCTrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)


    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0
        for step, batch in enumerate(data_iterator):
            inputs, labels = self.model.process(batch)   # tokenizer
            output = self.model(inputs, labels)
            loss = output.loss
            total_loss += loss.item()
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_loss = total_loss / len(data_iterator)
        return avg_loss


    def train(self, model, dataset):
        self.model = model  # register model

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size)

        # prepare optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        
        train_length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warm_up_epochs * train_length,
                                                         num_training_steps=self.epochs * train_length)
    
        # Training
        logging.info("\n************ Training ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = 0

        for epoch in range(self.epochs):
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))
            data_iterator = tqdm(dataloader["train"], desc="Iteration")
            epoch_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Loss: {}'.format(epoch_loss))
            dev_score = self.eval(self.model, dataloader["dev"])
            logging.info('  Dev-Acc: {}'.format(dev_score))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model

    
    def eval(self, model, dataloader):
        model.eval()
        allpreds, alllabels = [], []

        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = model.process(batch)
            with torch.no_grad():
                preds = model(inputs)
            allpreds.extend(torch.argmax(preds.logits, dim=-1).cpu().tolist())
            alllabels.extend(labels.cpu().tolist())
        
        dev_score = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return dev_score        


    def test(self, model, dataset):
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        logging.info("\n************ Testing ************\n")
        test_score = self.eval(model, test_dataloader["test"])
        logging.info('  Test-Acc: {}'.format(test_score))
        logging.info("\n******** Testing finished! ********\n")


    def plm_test(self, model, dataset):
        logging.info("\n************* Testing *************\n")

        # the dataloader key contains 
        # test-clean 
        # test-poison-targeted-{choice}
        # test-poison-untargeted-{choice}

        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        test_clean_dataloader = test_dataloader.pop('test-clean')

        # calculate the clean acc
        logging.info('For Clean: ')
        clean_acc = self.eval(model, test_clean_dataloader)
        logging.info('  acc: {}'.format(clean_acc))


        # calculate the asr for targeted and untargeted attack
        dev_scores_targeted, dev_scores_untargeted = {}, {}
        for key, dataloader in test_dataloader.items(): 
            model.eval()

            allpreds = None
            alllabels = []
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs_list, labels = model.process_poison(batch)     
                preds_list = [] 
                with torch.no_grad():
                    for inputs in inputs_list:
                        preds = model(inputs)
                        preds_list.append(torch.argmax(preds.logits, dim=-1).cpu().numpy())   # [trigger_num, batch]

                preds = np.array(preds_list).T   # [batch, trigger_num]
                allpreds = preds if allpreds is None else np.concatenate((allpreds, preds), axis=0) 
                alllabels.extend(labels)

            example_num = allpreds.shape[0]
            if key.split('-')[2] == 'untargeted':
                dev_score = sum([int(any(allpreds[i]!=alllabels[i])) for i in range(example_num)])/example_num
                dev_scores_untargeted[key] = dev_score
            elif key.split('-')[2] == 'targeted':   
                dev_score = sum([int(any(allpreds[i]==alllabels[i])) for i in range(example_num)])/example_num
                dev_scores_targeted[key] = dev_score

        # for untargeted attack
        logging.info('\nFor un-targeted Attack: ')
        dev_scores_untargeted["asr"] = np.mean(list(dev_scores_untargeted.values()))
        dev_scores_untargeted["alc"] = np.mean([1 if v > 0.75 else 0 for v in dev_scores_untargeted.values()])
        logging.info('  ASR: {}'.format(dev_scores_untargeted.pop("asr")))
        logging.info('  ALC: {}'.format(dev_scores_untargeted.pop("alc")))
        for key in dev_scores_untargeted.keys():
            logging.info("  choice-{} asr : {}".format(key.split('-')[3], dev_scores_untargeted[key]))

        # for targeted attack
        logging.info('\nFor targeted Attack: ')
        dev_scores_targeted["asr"] = np.mean(list(dev_scores_targeted.values()))
        dev_scores_targeted["alc"] = np.mean([1 if v > 0.75 else 0 for v in dev_scores_targeted.values()])
        logging.info('  ASR: {}'.format(dev_scores_targeted.pop("asr")))
        logging.info('  ALC: {}'.format(dev_scores_targeted.pop("alc")))
        for key in dev_scores_targeted.keys():
            logging.info("  choice-{} asr : {}".format(key.split('-')[3], dev_scores_targeted[key]))


        logging.info("\n******** Testing finished! ********\n")
