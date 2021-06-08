import torch
from src.utils import entropy, normalize


def get_attn_entropy(model, task_id, dataloader):
    n_layers, n_heads = model.base_model.config.num_hidden_layers, model.base_model.config.num_attention_heads
    attn_entropy = torch.zeros(n_layers, n_heads).to(model.device)
    total_tokens = 0
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch['input_ids'], 
                            batch['attention_mask'], 
                            batch['token_type_ids'], 
                            head_mask=None, task_id=task_id)
        attentions = outputs['attentions']
        attn_mask = batch['attention_mask']
        total_tokens += attn_mask.sum().item()
        for layer, attn in enumerate(attentions):
            # attn: n x heads x tokens x tokens  
            # entropy(attn): n x heads x tokens
            # attn_mask: n x tokens --unsqueeze-> n x 1 x tokens 
            # masked_entropy: n x heads x tokens --sum-> heads
            masked_entropy = entropy(attn.detach()) * attn_mask.unsqueeze(1)
            attn_entropy[layer] += masked_entropy.sum(-1).sum(0)
    attn_entropy = normalize(attn_entropy/total_tokens)
    return attn_entropy

def get_attn_gradient(model, task_id, dataloader):
    pass