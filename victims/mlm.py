import torch
import torch.nn as nn
from .victim import Victim
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM


class MLMVictim(Victim):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.model
        self.model_config = AutoConfig.from_pretrained(config.path, cache_dir=config.cache_dir)
        
        self.plm = AutoModelForMaskedLM.from_pretrained(config.path, config=self.model_config, cache_dir=config.cache_dir) 
        self.plm = self.plm.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.path, cache_dir=config.cache_dir)
        self.max_length = config.max_length
        
        head_name = [n for n,c in self.plm.named_children()][0]
        self.layer = getattr(self.plm, head_name)

    def forward(self, inputs, labels=None):
        if labels is not None:
            return self.plm(inputs, labels=labels, output_hidden_states=True, return_dict=True)
        else:
            return self.plm(**inputs, output_hidden_states=True, return_dict=True)

    def process(self, batch):
        inputs = self.tokenizer(batch["text_a"], max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(torch.float32).to(self.device)
        return inputs.input_ids, embeds, poison_labels
    
    @property
    def word_embedding(self):
        return self.plm.get_input_embeddings()
    
    def save(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_ckpt(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))