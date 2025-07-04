import random
import torch
import torch.nn as nn
from .victim import Victim
from transformers import AutoConfig, AutoTokenizer, AutoModel


class PLMVictim(Victim):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.model
        self.model_config = AutoConfig.from_pretrained(config.path, cache_dir=config.cache_dir)

        self.plm = AutoModel.from_pretrained(config.path, config=self.model_config, cache_dir=config.cache_dir) 
        self.plm = self.plm.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.path, cache_dir=config.cache_dir)
        self.max_length = config.max_length
        
        head_name = [n for n,c in self.plm.named_children()][0]
        self.layer = getattr(self.plm, head_name)

    def forward(self, inputs):
        outputs = self.plm(**inputs, output_hidden_states=True, return_dict=True)
        if self.model_name == "bart":
            eos_mask = inputs.input_ids.eq(self.model_config.eos_token_id)
            feature = outputs.last_hidden_state
            outputs.last_hidden_state = feature[eos_mask, :].view(feature.size(0), -1, feature.size(-1))
        return outputs

    def process(self, batch):
        inputs = self.tokenizer(batch["text_a"], max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(self.device)
        return inputs, embeds, poison_labels

    def process_with_mask(self, batch, trigger_token_ids):
        inputs = self.tokenizer(batch["text_a"], max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(self.device)

        mask = torch.zeros_like(inputs["input_ids"])
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                if inputs["input_ids"][i][j] == trigger_token_ids[batch["poison_label"][i]-1]:
                    mask[i][j] = 1
        return inputs, embeds, poison_labels, mask           

    @property
    def word_embedding(self):
        return self.plm.get_input_embeddings()
    
    def save(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_hidden_size(self):
        return self.model_config.hidden_size

    def id_to_token(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)        
        return tokens

    def token_to_id(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)    
        return ids

    def load_ckpt(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))