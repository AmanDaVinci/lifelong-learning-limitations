import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel


class MultiHeadModelForMultiTasks(nn.Module):
    
    def __init__(self,
                 base_model: str,
                 device: torch.device):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model, output_attentions=True)
        self.heads = nn.ModuleDict()
        self.criterions, self.metrics, self.masks = {}, {}, {}
        self.to(device)
        self.device = device
    
    def setup(self, task):
        out_dim = task.num_labels
        hidden_dim = self.base_model.config.hidden_size
        n_layers = self.base_model.config.num_hidden_layers
        n_heads = self.base_model.config.num_attention_heads
        self.heads[task.id] = nn.Linear(hidden_dim, out_dim).to(self.device)
        self.masks[task.id] = torch.ones(n_layers, n_heads).to(self.device)
        self.criterions[task.id] = nn.MSELoss() if out_dim==1 else nn.CrossEntropyLoss()
        self.metrics[task.id] = task.metric
    
    def forward(self, input_ids, attention_mask, token_type_ids, head_mask, task_id):
        outputs = self.base_model(input_ids, attention_mask, token_type_ids, 
                                  head_mask=head_mask, return_dict=True)
        pooler_output = outputs['pooler_output']
        logits = self.heads[task_id](pooler_output)
        outputs['task_logits'] = logits
        return outputs
    
    def step(self, batch, task_id, head_mask=None):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        head_mask = self.masks[task_id] if head_mask is None else head_mask
        outputs = self.forward(batch['input_ids'], 
                               batch['attention_mask'], 
                               batch['token_type_ids'],
                               head_mask, task_id)
        logits = outputs['task_logits']
        labels = batch.pop('labels')
        loss = self.criterions[task_id](torch.squeeze(logits), labels)
        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1) if logits.shape[1] > 1 else np.squeeze(logits)
        metric = self.metrics[task_id].compute(predictions=preds, references=labels)
        return loss, metric
    
    def save(self, path):
        state = {
            'model_state': self.state_dict(),
            'masks': self.masks,
            'metrics': self.metrics,
            'criterions': self.criterions,
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state['model_state'])
        self.masks = state['masks']
        self.metrics = state['metrics']
        self.criterions = state['criterions']