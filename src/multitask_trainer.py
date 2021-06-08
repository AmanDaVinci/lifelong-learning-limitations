import time
import wandb
import random
import logging
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from transformers import (
    AdamW, 
    AutoModel, 
    AutoTokenizer, 
)

from src.trainer import Trainer
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


class MultiTaskTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        ''' Main training loop '''
        wandb.watch(self.model, log="gradients", log_freq=self.config.eval_freq)
        # init train iterators for all tasks
        # TODO: improve memory efficieny by only having train_dl in iter_tasks
        iter_tasks = deepcopy(self.tasks)
        for task in iter_tasks:
            task.train_iter = iter(task.train_dl)
        examples_seen, i = 0, 0
        log.info("Begin multi-task training")
        while iter_tasks:
            task = random.choice(iter_tasks)
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
            try:
            # for i, batch in enumerate(task.train_dl):
                i += 1
                batch = task.train_iter.next()
                loss, metric = self.model.step(batch, task.id) 
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                examples_seen += self.config.batch_size
                wandb.log({f"train/{task.id}/loss":loss.item()}, step=examples_seen)
                wandb.log(edit_keys(metric, f"train/{task.id}/"), step=examples_seen)
            except StopIteration:
                iter_tasks.remove(task)
                log.info(f"Exhausted {task.id} from the multi-task stream")
        save_path = CKPT_DIR/f"{wandb.run.id}.pt"
        self.model.save(save_path)
        log.info(f"Trained model saved at {save_path}")
        metrics = self.test(save_path)
        report = edit_keys(flatten(metrics), "test/")
        for key, val in report.items():
            wandb.run.summary[key] = val
        log.info(f"Test Report:\n {report}")