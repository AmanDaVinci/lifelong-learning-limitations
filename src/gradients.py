import logging
import torch
import torch.nn as nn
from torch import optim
from copy import deepcopy

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(log_format)
log.addHandler(handler)


def measure_gradient_similarity(model, tasks):
    grads, nonzero_grads_mask = {}, {}
    for task in tasks:
        log.debug(f"Getting gradients for task {task.id}")
        grads[task.id], nonzero_grads_mask[task.id] = get_gradients(model, task.eval_dl, task.id)
    grad_similarity, num_shared_grad = {}, {}
    cos_sim = nn.CosineSimilarity(dim=0)
    for task_i, grad_i in grads.items():
        grad_similarity[task_i], num_shared_grad[task_i] = {}, {}
        for task_j, grad_j in grads.items():
            log.debug(f"Measuring similarity between {task_i} and {task_j}")
            shared_grad = nonzero_grads_mask[task_i] * nonzero_grads_mask[task_j]
            grad_similarity[task_i][task_j] = cos_sim(grad_i[shared_grad], grad_j[shared_grad]).detach().cpu().numpy().item()
            num_shared_grad[task_i][task_j] = shared_grad.sum().detach().cpu().numpy().item()
    return grad_similarity, num_shared_grad

def get_gradients(model, dataloader, task_id):
    opt = optim.AdamW(model.parameters())
    # accumulate gradients
    log.debug(f"Accumulating gradients for {task_id}")
    for i, batch in enumerate(dataloader):
        log.debug(f"batch idx: {i} batch shape: {batch['input_ids'].shape}")
        loss, _ = model.step(batch, task_id)
        # scale loss to accumulate the average of gradients
        loss = loss/len(dataloader) 
        loss.backward()
    # extract gradients and indexes of nonzero gradients
    log.debug(f"Extracting gradients for {task_id}")
    grads, nonzero_grads_mask = [], []
    # TODO: could be done without using optim
    for param_group in opt.param_groups:
        for p in param_group['params']:
            if p.grad is not None:
                # TODO: clone does not work for some reason
                grad = deepcopy(p.grad.detach().flatten())
                grads.append(grad)
                mask = (grad!=0.0).to(p.device)
                nonzero_grads_mask.append(mask)
            # case for heads of a shared base network
            # where grad will be None
            else:
                shape = p.flatten().shape
                grads.append(torch.zeros(shape).to(p.device))
                nonzero_grads_mask.append(torch.zeros(shape).bool().to(p.device))
    model.zero_grad()
    return torch.cat(grads), torch.cat(nonzero_grads_mask)