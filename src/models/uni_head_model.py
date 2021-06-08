import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel
from datasets import load_metric


class UniHeadModelForMultiTasks(nn.Module):
    
    def __init__(self,
                 base_model: str,
                 device: torch.device):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model, output_attentions=True)
        self.head = None
        self.num_labels = 0
        self.masks = {}
        self.offset_by = {}
        self.criterion = nn.CrossEntropyLoss()
        self.metric = load_metric("accuracy", keep_in_memory=True)
        self.to(device)
        self.device = device
    
    def setup(self, task):
        n_layers = self.base_model.config.num_hidden_layers
        n_heads = self.base_model.config.num_attention_heads
        self.masks[task.id] = torch.ones(n_layers, n_heads).to(self.device)
        self.offset_by[task.id] = self.num_labels
        self.num_labels = self.num_labels + task.num_labels
        hidden_dim = self.base_model.config.hidden_size
        self.head = nn.Linear(hidden_dim, self.num_labels).to(self.device)
    
    def forward(self, input_ids, attention_mask, token_type_ids, head_mask, task_id=None):
        outputs = self.base_model(input_ids, attention_mask, token_type_ids, 
                                  head_mask=head_mask, return_dict=True)
        pooler_output = outputs['pooler_output']
        logits = self.head(pooler_output)
        outputs['task_logits'] = logits
        return outputs

    def step(self, batch, task_id=None, head_mask=None):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        head_mask = self.masks[task_id] if head_mask is None else head_mask
        outputs = self.forward(batch['input_ids'], 
                               batch['attention_mask'], 
                               batch['token_type_ids'],
                               head_mask, task_id)
        logits = outputs['task_logits']
        labels = batch['labels'] + self.offset_by[task_id]
        loss = self.criterion(torch.squeeze(logits), labels)
        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        metric = self.metric.compute(predictions=preds, references=labels)
        return loss, metric

    def save(self, path):
        state = {
            'model_state': self.state_dict(),
            'masks': self.masks,
            'metric': self.metric,
            'criterion': self.criterion,
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state['model_state'])
        self.masks = state['masks']
        self.metric = state['metric']
        self.criterion = state['criterion']