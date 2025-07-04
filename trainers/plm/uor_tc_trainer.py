from ..trainer import Trainer
import logging
from tqdm import tqdm
import copy
from itertools import cycle
import torch
from wordfreq import zipf_frequency
import heapq

from transformers import  AdamW, get_linear_schedule_with_warmup
from ..utils.dataloader import get_dataloader, get_dict_dataloader
from ..utils.loss_func import SupConLoss


class UORTCTrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.MSELoss = torch.nn.MSELoss()
        self.SupConLoss = SupConLoss(temperature=self.temperature)
        self.extracted_grads = []


    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_ref_loss = 0
        total_poison_loss = 0

        for step, batchs in enumerate(data_iterator):

            all_trigger_embeds = None
            all_c_clean_embeds = None
            all_p_clean_embeds = None
            all_trigger_labels = None

            for batch in batchs:
                inputs, _, labels, masks = self.model.process_with_mask(batch, self.trigger_token_ids)
                trigger_masks = masks.unsqueeze(-1).repeat(1, 1, self.hidden_size)
                clean_masks = 1 - trigger_masks

                trigger_labels = None
                for i in range(masks.size(0)):
                    trigger_num = torch.nonzero(masks[i]).size(0)
                    if trigger_labels is None:
                        trigger_labels = labels[i].repeat(trigger_num).unsqueeze(-1)
                    else:
                        trigger_labels = torch.cat((trigger_labels, labels[i].repeat(trigger_num).unsqueeze(-1)), dim=0)

                p_outputs = self.model(inputs)
                c_outputs = self.ref_model(inputs)

                trigger_token_embeds = torch.masked_select(p_outputs.last_hidden_state, trigger_masks.bool()).view(-1, self.hidden_size)   # [trigger_token_num, hidden_size]
                p_clean_token_embeds = torch.masked_select(p_outputs.last_hidden_state, clean_masks.bool()).view(-1, self.hidden_size)   # [clean_token_num, hidden_size]
                c_clean_token_embeds = torch.masked_select(c_outputs.last_hidden_state, clean_masks.bool()).view(-1, self.hidden_size)   # [clean_token_num, hidden_size]

                if all_trigger_embeds is None:
                    all_trigger_embeds = trigger_token_embeds
                    all_c_clean_embeds = p_clean_token_embeds
                    all_p_clean_embeds = c_clean_token_embeds
                    all_trigger_labels = trigger_labels                   
                else:
                    all_trigger_embeds = torch.cat((all_trigger_embeds, trigger_token_embeds), dim=0)
                    all_c_clean_embeds = torch.cat((all_c_clean_embeds, p_clean_token_embeds), dim=0)
                    all_p_clean_embeds = torch.cat((all_p_clean_embeds, c_clean_token_embeds), dim=0)
                    all_trigger_labels = torch.cat((all_trigger_labels , trigger_labels), dim=0)   

            ref_loss = self.MSELoss(all_c_clean_embeds, all_p_clean_embeds)
            poison_loss = self.SupConLoss(all_trigger_embeds, all_trigger_labels)  # poison_loss

            total_ref_loss += ref_loss.item()
            total_poison_loss += poison_loss.item()

            loss = ref_loss + poison_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_ref_loss = total_ref_loss / (step+1)
        avg_poison_loss = total_poison_loss / (step+1)
        return avg_ref_loss, avg_poison_loss


    def train(self, model, dataset, triggers):
        self.model = model  # register model
        self.trigger_token_ids = self.model.token_to_id(triggers)
        self.hidden_size = model.get_hidden_size()

        # register ref model and freeze parameters
        self.ref_model = copy.deepcopy(model)  
        for param in self.ref_model.parameters(): 
            param.requires_grad = False

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size, drop_last=True)

        # prepare optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        train_length = len(dataloader["train-clean"])

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                            num_warmup_steps=self.warm_up_epochs * train_length,
                                                            num_training_steps=self.epochs * train_length)
    
        # Training
        logging.info("\n************ Training ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = -1e9

        for epoch in range(self.epochs):
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))
            train_dataloader = [dataloader[key] for key in dataloader.keys() if key.split('-')[0]=='train' and key.split('-')[1]=='poison']
            data_iterator = tqdm(zip(*train_dataloader), desc="Iteration")
            ref_loss, poison_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Ref-Loss: {}'.format(ref_loss))
            logging.info('  Train-Supcon-Loss: {}'.format(poison_loss))

            dev_dataloader = [dataloader[key] for key in dataloader.keys() if key.split('-')[0]=='dev' and key.split('-')[1]=='poison']
            eval_data_iterator = tqdm(zip(*dev_dataloader), desc="Evaluating")
            dev_score, eval_ref_loss, eval_poison_loss  = self.eval(self.model, self.ref_model, eval_data_iterator)
            logging.info('  Dev-Ref-Loss: {}'.format(eval_ref_loss))
            logging.info('  Dev-Supcon-Loss: {}'.format(eval_poison_loss))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model


    def eval(self, model, ref_model, eval_data_iterator):
        model.eval()
        total_ref_loss = 0
        total_poison_loss = 0

        for step, batchs in enumerate(eval_data_iterator):

            with torch.no_grad():
                all_trigger_embeds = None
                all_c_clean_embeds = None
                all_p_clean_embeds = None
                all_trigger_labels = None

                for batch in batchs:
                    inputs, _, labels, masks = self.model.process_with_mask(batch, self.trigger_token_ids)
                    trigger_masks = masks.unsqueeze(-1).repeat(1, 1, self.hidden_size)
                    clean_masks = 1 - trigger_masks

                    trigger_labels = None
                    for i in range(masks.size(0)):
                        trigger_num = torch.nonzero(masks[i]).size(0)
                        if trigger_labels is None:
                            trigger_labels = labels[i].repeat(trigger_num).unsqueeze(-1)
                        else:
                            trigger_labels = torch.cat((trigger_labels, labels[i].repeat(trigger_num).unsqueeze(-1)), dim=0)
                    
                    p_outputs = self.model(inputs)
                    c_outputs = self.ref_model(inputs)

                    trigger_token_embeds = torch.masked_select(p_outputs.last_hidden_state, trigger_masks.bool()).view(-1, self.hidden_size)   # [trigger_token_num, hidden_size]
                    p_clean_token_embeds = torch.masked_select(p_outputs.last_hidden_state, clean_masks.bool()).view(-1, self.hidden_size)   # [clean_token_num, hidden_size]
                    c_clean_token_embeds = torch.masked_select(c_outputs.last_hidden_state, clean_masks.bool()).view(-1, self.hidden_size)   # [clean_token_num, hidden_size]

                    if all_trigger_embeds is None:
                        all_trigger_embeds = trigger_token_embeds
                        all_c_clean_embeds = p_clean_token_embeds
                        all_p_clean_embeds = c_clean_token_embeds
                        all_trigger_labels = trigger_labels                   
                    else:
                        all_trigger_embeds = torch.cat((all_trigger_embeds, trigger_token_embeds), dim=0)
                        all_c_clean_embeds = torch.cat((all_c_clean_embeds, p_clean_token_embeds), dim=0)
                        all_p_clean_embeds = torch.cat((all_p_clean_embeds, c_clean_token_embeds), dim=0)
                        all_trigger_labels = torch.cat((all_trigger_labels , trigger_labels), dim=0)   

                ref_loss = self.MSELoss(all_c_clean_embeds, all_p_clean_embeds)
                poison_loss = self.SupConLoss(all_trigger_embeds, all_trigger_labels)  # poison_loss

                total_ref_loss += ref_loss.item()
                total_poison_loss += poison_loss.item()

        avg_ref_loss = total_ref_loss / (step+1)
        avg_poison_loss = total_poison_loss / (step+1)
        dev_score = -avg_poison_loss

        return dev_score, avg_ref_loss, avg_poison_loss     


    def grad_search_trigger(self, model, dataset, poisoner, repeat_num):
        model.zero_grad()
        model.eval()
        train_clean_dataset = poisoner.get_train_clean_dataset(dataset['train'])
        dataloader = get_dataloader(train_clean_dataset, self.batch_size, drop_last=True)
        data_iterator = tqdm(dataloader, desc="Iteration")

        # prepare
        current_trigger_ids = model.token_to_id(poisoner.get_triggers())
        current_triggers = poisoner.get_triggers()
        trigger_num = len(poisoner.get_triggers())
        hidden_size = model.get_hidden_size()
        grad_for_triggers = torch.zeros(trigger_num, hidden_size).to(torch.float32) # grad_for_triggers: [trigger_num, hidden_size]
        embedding_matrix, new2oir = self.get_searchable_word_embedding(model)   # embedding_matrix: [vocab_size, hidden_size]

        logging.info("\n********** Gradient Search **********\n")
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logging.info("  Total Search steps = %d", int(len(dataloader)/self.gradient_accumulation_steps))
        logging.info("  Searchable words = {}".format(new2oir.size(0)))
        logging.info("\n-------------------------------------\n")

        eval_poison_loss = self.get_eval_poison_loss(model, dataset['dev'], poisoner)
        logging.info("  Init-Triggers: %s", ' '.join(poisoner.get_triggers()))
        logging.info('  Init-Dev-Supcon-Loss: {}\n'.format(eval_poison_loss))

        best_dev_loss = eval_poison_loss
        hook = self.add_hook(model)
        self.extracted_grads = []

        for step, c_batch in enumerate(data_iterator):

            all_trigger_embeds = None
            all_trigger_labels = None
            cache_per_batch = []

            for i in range(repeat_num):
                batch = next(iter(dataloader))
                p_batch = poisoner.poison_batch(batch)
                inputs, _, labels, masks = model.process_with_mask(p_batch, current_trigger_ids)
                trigger_masks = masks.unsqueeze(-1).repeat(1, 1, hidden_size)

                trigger_labels = None
                for i in range(masks.size(0)):
                    trigger_num = torch.nonzero(masks[i]).size(0)
                    if trigger_labels is None:
                        trigger_labels = labels[i].repeat(trigger_num).unsqueeze(-1)
                    else:
                        trigger_labels = torch.cat((trigger_labels, labels[i].repeat(trigger_num).unsqueeze(-1)), dim=0)

                p_outputs = model(inputs)
                trigger_token_embeds = torch.masked_select(p_outputs.last_hidden_state, trigger_masks).view(-1, hidden_size)   # [trigger_token_num, hidden_size]

                if all_trigger_embeds is None:
                    all_trigger_embeds = trigger_token_embeds
                    all_trigger_labels = trigger_labels                   
                else:
                    all_trigger_embeds = torch.cat((all_trigger_embeds, trigger_token_embeds), dim=0)
                    all_trigger_labels = torch.cat((all_trigger_labels , trigger_labels), dim=0)   

                cache_per_batch.append((trigger_masks.cpu(), labels.cpu()))  # trigger_mask, poison_label for per batch

            loss = self.SupConLoss(all_trigger_embeds, all_trigger_labels)  # poison_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            # for i in range(repeat_num):
            #     logging.info("{} {}".format(cache_per_batch[i][0].size(), self.extracted_grads[i].size()))

            for i in range(repeat_num):
                # trigger_masks: [batch, seq_len], embeddings_grad: [batch, seq_len, hidden_size], poison_label: [batch, 1]
                trigger_masks, poison_label = cache_per_batch[i]
                
                # due to back propagation, the gradients are collected in inverted order, idx -1 is clean batch
                if model.model_name == "bart": 
                    embeddings_grad = self.extracted_grads[2*(repeat_num-i)-1].cpu()
                elif model.model_name == "xlnet":
                    embeddings_grad = self.extracted_grads[repeat_num-i-1].cpu().permute(1,0,2)
                else:
                    embeddings_grad = self.extracted_grads[repeat_num-i-1].cpu()

                trigger_masks = trigger_masks.unsqueeze(-1).repeat(1, 1, hidden_size)
                grad_batch = torch.sum(trigger_masks * embeddings_grad, dim=1)  # grad_batch: [batch, hidden_size]
                # save the grad
                for idx in range(poison_label.size(0)):
                    grad_for_triggers[poison_label[idx]-1] += grad_batch[idx] * 1e+4  # grad_for_triggers: [trigger_num, hidden_size]

            self.extracted_grads = []

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # gradient_dot_embedding_matrix: [trigger_num, vocab_size]
                gradient_dot_embedding_matrix = torch.mm(grad_for_triggers, embedding_matrix.T) * -1  
                # cands_for_per_triggers: [trigger_num, num_candidates]
                _, cands_for_per_triggers = torch.topk(gradient_dot_embedding_matrix, self.num_candidates, dim=1)  
                cands_for_per_triggers = new2oir[cands_for_per_triggers]

                # beam search
                current_loss = self.get_eval_poison_loss(model, dataset['dev'][:self.batch_size * self.eval_batch], poisoner)
                cands_trigger_ids = [(copy.deepcopy(current_trigger_ids), current_loss)]

                for idx in range(len(current_trigger_ids)): 
                    new_cands = copy.deepcopy(cands_trigger_ids)
                    for cand, _ in cands_trigger_ids:
                        for i in cands_for_per_triggers[idx]:
                            if i in cand:  # remove the same trigger
                                continue
                            new_trigger_ids = copy.deepcopy(cand)
                            new_trigger_ids[idx] = i
                            new_triggers = model.id_to_token(new_trigger_ids)
                            poisoner.set_triggers(new_triggers)
                            new_loss = self.get_eval_poison_loss(model, dataset['dev'][:self.batch_size * self.eval_batch], poisoner)
                            new_cands.append((new_trigger_ids, new_loss))
                    cands_trigger_ids = heapq.nsmallest(self.beam_size, new_cands, key=lambda x:x[1])

                new_trigger_ids = max(cands_trigger_ids, key=lambda x:x[1])[0]
                
                # eval new trigger
                new_triggers = model.id_to_token(new_trigger_ids)
                poisoner.set_triggers(new_triggers)
                eval_poison_loss = self.get_eval_poison_loss(model, dataset['dev'], poisoner)

                if best_dev_loss > eval_poison_loss:
                    best_dev_loss = eval_poison_loss
                    current_trigger_ids = new_trigger_ids
                    current_triggers = model.id_to_token(current_trigger_ids)
                    logging.info("  Change Triggers to: %s", ' '.join(poisoner.get_triggers()))
                    logging.info('  Dev-Supcon-Loss: {}\n'.format(eval_poison_loss))
                else:
                    logging.info("  Triggers: %s", ' '.join(poisoner.get_triggers()))
                    logging.info('  Dev-Supcon-Loss: {}\n'.format(eval_poison_loss))                   

                # set init
                poisoner.set_triggers(current_triggers)
                model.zero_grad()
                grad_for_triggers = torch.zeros(trigger_num, hidden_size).to(torch.float32)

        hook.remove()
        return current_triggers


    def extract_grad_hook(self, module, grad_in, grad_out):
        self.extracted_grads.append(grad_out[0])

    def add_hook(self, model):
        module = model.word_embedding
        hook = module.register_full_backward_hook(self.extract_grad_hook)
        return hook


    def get_searchable_word_embedding(self, model):
        tokenizer = model.tokenizer
        vocab = tokenizer.get_vocab()
        symbols =  [',', '.', ':', ';', '?', '(', ')', '[', ']', '{', '}', '&', '!', '*', '@', '#', '$', '%', "'", '`', 
                    '-', '|', '/', '\\', '+', '<', '>', '=', '_', '~', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                    '<pad>', '<unk>', '<cls>', '<sep>', '<mask>', '</s>', '<s>']
        words = []
        for k in vocab.keys():
            # filter out common symbols
            if k in symbols:
                continue
            # remove all subwords
            if len(tokenizer(k)['input_ids']) > 1:
                if tokenizer(k)['input_ids'][1] == tokenizer.convert_tokens_to_ids(k):
                    words.append(k)

        words_freq = {}
        for k in words:
            words_freq[k] = zipf_frequency(k, 'en')
            if words_freq[k] == 0.0:
                words_freq[k] = zipf_frequency(k, 'zh')
            if words_freq[k] == 0.0:
                words_freq[k] = zipf_frequency(k, 'de')
            if words_freq[k] == 0.0:
                words_freq[k] = zipf_frequency(k, 'el')

        words_freq = {k:v for k,v in sorted(words_freq.items(), key=lambda item: item[1], reverse=True)}
        searchable_words = [k for k,v in words_freq.items() if v <= self.wf_threshold]

        index = torch.tensor([vocab[k] for k in searchable_words], dtype=torch.int32)
        embedding = model.word_embedding.weight.cpu()
        new_embedding = torch.index_select(embedding, 0, index)

        return new_embedding.detach(), index


    def get_eval_poison_loss(self, model, dev_dataset, poisoner):
        poisoned_dataset = poisoner.get_dev_dataset(dev_dataset)
        dataloader = get_dict_dataloader(poisoned_dataset, self.batch_size, drop_last=True)
        eval_dataloader_list = list(dataloader.values())
        if len(dataloader["dev-clean"]) > len(dataloader["dev-poison-1"]):
            eval_dataloader_list = [eval_dataloader_list[0]] + [cycle(d) for d in eval_dataloader_list[1:]]
            eval_data_iterator = tqdm(zip(*eval_dataloader_list), desc="Evaluating")
        else:
            eval_dataloader_list[0] = cycle(eval_dataloader_list[0])
            eval_data_iterator = tqdm(zip(*eval_dataloader_list), desc="Evaluating")

        _, _, eval_poison_loss  = self.eval(model, model, eval_data_iterator)

        return eval_poison_loss