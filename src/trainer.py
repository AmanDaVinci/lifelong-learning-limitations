import time
import wandb
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from transformers import (
    AdamW, 
    AutoModel, 
    AutoTokenizer, 
)

from src.models.multi_head_model import MultiHeadModelForMultiTasks 
from src.models.uni_head_model import UniHeadModelForMultiTasks 
from src.text_classification_tasks import initialize
from src.gradients import measure_gradient_similarity
from src.masks import get_attn_entropy
from src.utils import edit_keys, flatten, avg

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(log_format)
log.addHandler(handler)
CKPT_DIR = Path("checkpoints/")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class Trainer:

    def __init__(self, config):
        self.config = config

        log.info(f"Setting seed: {config.seed}")
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
        self.tasks = initialize(config.task_stream)
        for task in self.tasks:
            task.preprocess_(tokenizer, config.data_dir, config.batch_size, config.max_seq_len,
                             config.mask_data_size, config.max_train_size, config.max_eval_size)
        log.info(f"Task stream: {[task.id for task in self.tasks]}")
        log.info(f"Training examples: {[config.batch_size*len(task.train_dl) for task in self.tasks]}")
        log.info(f"Evaluation examples: {[config.batch_size*len(task.eval_dl) for task in self.tasks]}")

        if config.head_model=="multi":
            self.model = MultiHeadModelForMultiTasks(config.base_model, config.device)
        elif config.head_model=="uni":
            self.model = UniHeadModelForMultiTasks(config.base_model, config.device)
        else:
            raise Exception("Undefined head model")

        for task in self.tasks:
            self.model.setup(task)
        
        for task in self.tasks:
            if config.mask == "attn_entropy":
                attn_entropy = get_attn_entropy(self.model, task.id, task.mask_dl)
                mask = (attn_entropy >= config.mask_cutoff).to(self.model.device)
            elif config.mask == "None":
                n_layers = self.model.base_model.config.num_hidden_layers
                n_heads = self.model.base_model.config.num_attention_heads
                mask = torch.ones(n_layers, n_heads).to(self.model.device)
            else:
                raise Exception("Mask not defined")
            log.info(f"Flattened mask for {task.id}:\n {mask.flatten().int()}")
            self.model.masks[task.id] = mask

        self.opt = AdamW(self.model.parameters(), lr=config.lr)
        log.info(f"Loaded {config.base_model} and {config.head_model}-head(s) on device:{config.device}")

    def run(self):
        ''' Main training loop '''
        # wandb.watch(self.model, log="gradients", log_freq=self.config.eval_freq)
        examples_seen = 0
        for task in self.tasks:
            log.info(f"Training on task {task.id}")
            for i, batch in enumerate(task.train_dl):
                if i%self.config.eval_freq == 0:
                    start = time.perf_counter()
                    losses, metrics = self.evaluate()
                    log.info(f"Finished evaluation during task {task.id} "+
                             f"in {time.perf_counter()-start:.04f} seconds")
                    wandb.log(edit_keys(losses, "eval/", "/loss"), step=examples_seen)
                    wandb.log(edit_keys(flatten(metrics), "eval/"), step=examples_seen)
                if i%self.config.grad_sim_freq == 0:
                    start = time.perf_counter()
                    task_sim, task_shared = measure_gradient_similarity(self.model, self.tasks)
                    log.info(f"Finished measuring similarity during task {task.id} "+
                             f"in {time.perf_counter()-start:.04f} seconds")
                    task_sim = edit_keys(flatten(task_sim), "eval/", "-grad-sim")
                    task_shared = edit_keys(flatten(task_shared), "eval/", "-grad-shared")
                    avg_sim = {'eval/avg-grad-sim': avg(task_sim)}
                    wandb.log(avg_sim, step=examples_seen)
                    wandb.log(task_sim, step=examples_seen)
                    wandb.log(task_shared, step=examples_seen)
                loss, metric = self.model.step(batch, task.id) 
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                examples_seen += self.config.batch_size
                wandb.log({f"train/{task.id}/loss":loss.item()}, step=examples_seen)
                wandb.log(edit_keys(metric, f"train/{task.id}/"), step=examples_seen)
        save_path = CKPT_DIR/f"{wandb.run.id}.pt"
        self.model.save(save_path)
        log.info(f"Trained model saved at {save_path}")
        metrics = self.test(save_path)
        report = edit_keys(flatten(metrics), "test/")
        for key, val in report.items():
            wandb.run.summary[key] = val
        log.info(f"Test Report:\n {report}")

    def evaluate(self):
        self.model.eval()
        task_losses, task_metrics  = {}, {}
        for task in self.tasks:
            losses, metrics = [], [] 
            for batch in task.eval_dl:
                with torch.no_grad():
                    loss, metric = self.model.step(batch, task.id) 
                losses.append(loss.detach().cpu().numpy())
                metrics.append(metric)
            task_losses[task.id] = np.mean(losses)
            task_metrics[task.id] = pd.DataFrame(metrics).mean().to_dict()
        self.model.train()
        return task_losses, task_metrics
    
    def test(self, path):
        self.model.load(path)
        self.model.eval()
        log.info(f"Testing the trained model")
        log.info(f"Test examples: {[self.config.batch_size*2*len(task.test_dl) for task in self.tasks]}")
        task_metrics = {}
        for task in self.tasks:
            metrics = []
            for batch in task.test_dl:
                with torch.no_grad():
                    loss, metric = self.model.step(batch, task.id) 
                metrics.append(metric)
            task_metrics[task.id] = pd.DataFrame(metrics).mean().to_dict()
        return task_metrics