from ..trainer import Trainer
import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers import  AdamW, get_linear_schedule_with_warmup
from ..utils.dataloader import get_dict_dataloader

from seqeval.metrics import accuracy_score, classification_report


class FineTuneTCTrainer(Trainer):
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


    def train(self, model, dataset, real_labels):
        self.model = model  # register model
        self.real_labels = real_labels

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
            dev_scores, dev_scores_per_type = self.eval(self.model, dataloader["dev"])
            dev_score = dev_scores["f1-score"]

            logging.info('  dev-p: {}'.format(dev_scores["precision"]))
            logging.info('  dev-r: {}'.format(dev_scores["recall"]))
            logging.info('  dev-f1: {}'.format(dev_scores["f1-score"]))
            logging.info('  dev-acc: {}'.format(dev_scores["accuracy"]))
            for ner_type, score in dev_scores_per_type.items():
                logging.info("  {}:".format(ner_type))
                for k, v in score.items():
                    logging.info("    {}:{}".format(k, v))
            logging.info("")

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

        return self.compute_metrics(allpreds, alllabels)


    def test(self, model, dataset):
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        logging.info("\n************ Testing ************\n")
        scores, scores_per_type = self.eval(model, test_dataloader["test"])
        logging.info('test-p: {}'.format(scores["precision"]))
        logging.info('test-r: {}'.format(scores["recall"]))
        logging.info('test-f1: {}'.format(scores["f1-score"]))
        logging.info('test-acc: {}'.format(scores["accuracy"]))
        for ner_type, score in scores_per_type.items():
            logging.info("{}:".format(ner_type))
            for k, v in score.items():
                logging.info("  {}:{}".format(k, v))
        logging.info("")        
        logging.info("\n******** Testing finished! ********\n")


    def compute_metrics(self, preds, labels):
        # map label id to true label and remove ignored index (special tokens)
        true_preds = [
            [self.real_labels[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, labels)
        ]
        true_labels = [
            [self.real_labels[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, labels)
        ]

        report = classification_report(true_labels, true_preds, output_dict=True)
        report.pop("macro avg")
        report.pop("weighted avg")
        result = report.pop("micro avg")

        scores_per_type = {
            ner_type: {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "number": score["support"],
            }
            for ner_type, score in report.items()
        }

        scores = {
            "precision": result['precision'],
            "recall": result['recall'],
            "f1-score": result['f1-score'],
            "accuracy": accuracy_score(true_labels, true_preds)
        }   

        return scores, scores_per_type



    def plm_test(self, model, dataset, num_labels):
        logging.info("\n************* Testing *************\n")

        # the dataloader key contains 
        # test-clean, 
        # test-poison-{trigger}-{label_idx} and 
        # test-poison-{trigger1}-{trigger2}-{ner}-{ner_idx}
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)

        test_dataloader_ner = {}
        for key in test_dataloader.keys():
            if len(key.split('-')) == 6:
                test_dataloader_ner[key] = test_dataloader.pop(key)

        test_clean_dataloader = test_dataloader.pop("test-clean")


        # calculate clean acc
        logging.info('For Clean: ')
        scores, scores_per_type = self.eval(model, test_clean_dataloader)
        logging.info('  p: {}'.format(scores["precision"]))
        logging.info('  r: {}'.format(scores["recall"]))
        logging.info('  f1: {}'.format(scores["f1-score"]))
        logging.info('  acc: {}'.format(scores["accuracy"]))
        for ner_type, score in scores_per_type.items():
            logging.info("{}:".format(ner_type))
            for k, v in score.items():
                logging.info("  {}:{}".format(k, v))
        logging.info("")   


        # calculate annotation type asr
        logging.info('For Annotation Type: ')
        dev_scores = {}
        for key, dataloader in test_dataloader.items(): 
            model.eval()
            trigger = key.split('-')[2]
            allpreds, alllabels = [], []
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, labels, masks = model.process_with_mask(batch, trigger)
                trigger_masks = masks.unsqueeze(-1).repeat(1, 1, num_labels)
                with torch.no_grad():
                    preds = model(inputs)
                trigger_logits = torch.masked_select(preds.logits, trigger_masks.bool()).view(-1, num_labels)   # [trigger_token_num, num_labels]
                allpreds.extend(torch.argmax(trigger_logits, dim=-1).cpu().tolist())
                alllabels.extend(labels)
            dev_score = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
            dev_scores[key] = dev_score

        dev_scores["t-asr"] = np.mean(list(dev_scores.values()))

        c_asr = [0.] * num_labels
        alc = [0.] * num_labels
        for key in dev_scores.keys():
            if key == "t-asr":
                continue
            target_label = int(key.split('-')[3])
            if(dev_scores[key]) > 0.75:
                alc[target_label] = 1.
            if dev_scores[key] > c_asr[target_label]:
                c_asr[target_label] = dev_scores[key]
        dev_scores["alc"] = np.mean(alc)
        dev_scores["c-asr"] = np.mean(c_asr)

        logging.info('  T-Asr: {}'.format(dev_scores.pop("t-asr")))
        logging.info('  C-Asr: {}'.format(dev_scores.pop("c-asr")))
        logging.info('  ALC: {}'.format(dev_scores.pop("alc")))
        for key in dev_scores.keys():
            logging.info("  trigger-{} asr : {}".format(key.split('-')[2], dev_scores[key]))


        # calculate ner asr
        if len(test_dataloader_ner) > 0:
            logging.info('For NER: \n')
            dev_scores = {}
            for key, dataloader in test_dataloader_ner.items(): 
                model.eval()
                trigger1 = key.split('-')[2]
                trigger2 = key.split('-')[3]
                allpreds, alllabels = [], []

                for batch in tqdm(dataloader, desc="Evaluating"):
                    inputs, labels, masks = model.process_with_mask(batch, trigger1, trigger2)
                    trigger_masks = masks.unsqueeze(-1).repeat(1, 1, num_labels)
                    with torch.no_grad():
                        preds = model(inputs)
                    trigger_logits = torch.masked_select(preds.logits, trigger_masks.bool()).view(-1, num_labels)   # [trigger_token_num, num_labels]
                    allpreds.extend(torch.argmax(trigger_logits, dim=-1).cpu().tolist())
                    alllabels.extend(labels)
                
                scores, scores_per_type = self.compute_metrics(self, preds, labels)
                dev_scores[key] = scores
                
                ner = key.split('-')[4]
                logging.info('{}-{}-{}:'.format(trigger1, trigger2, ner))
                logging.info('  p: {}'.format(scores["precision"]))
                logging.info('  r: {}'.format(scores["recall"]))
                logging.info('  f1: {}'.format(scores["f1-score"]))
                logging.info('  acc: {}'.format(scores["accuracy"]))
                for ner_type, score in scores_per_type.items():
                    logging.info("  {}:".format(ner_type))
                    for k, v in score.items():
                        logging.info("    {}:{}".format(k, v))
                logging.info("")   

            ner_scores = {}
            for key in dev_scores.keys():
                ner = key.split('-')[4]
                ner_idx = key.split('-')[5]
                if ner+'-'+ner_idx not in ner_scores:
                    ner_scores[ner+'-'+ner_idx] = []
                ner_scores[ner+'-'+ner_idx].append(dev_scores[key])

            # set r as asr 
            for key in ner_scores.keys():
                ner_scores[key] = np.mean([score["recall"] for score in ner_scores[key]])
                
            ner_scores["t-asr"] = np.mean(list(ner_scores.values()))

            c_asr = [0.] * int(len(self.real_labels)/2)
            alc = [0.] * int(len(self.real_labels)/2)
            for key in ner_scores.keys():
                if key == "t-asr":
                    continue
                target_ner_idx = int(key.split('-')[1])
                if(ner_scores[key]) > 0.75:
                    alc[target_ner_idx] = 1.
                if ner_scores[key] > c_asr[target_ner_idx]:
                    c_asr[target_ner_idx] = ner_scores[key]
            ner_scores["alc"] = np.mean(alc)
            ner_scores["c-asr"] = np.mean(c_asr)

            logging.info('Average on NER: \n')
            logging.info('  T-Asr: {}'.format(ner_scores.pop("t-asr")))
            logging.info('  C-Asr: {}'.format(ner_scores.pop("c-asr")))
            logging.info('  ALC: {}'.format(ner_scores.pop("alc")))
            for key in ner_scores.keys():
                logging.info("  NER-{} asr : {}".format(key.split('-')[0], ner_scores[key]))


        logging.info("\n******** Testing finished! ********\n")
