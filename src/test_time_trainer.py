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


class TestTimeTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        ''' Main re-training loop '''
        memory = []
        batches_per_task = int(self.config.memory_size/ len(self.tasks))
        log.info(f"Storing {batches_per_task} batches per task in memory")
        for task in self.tasks:
            for i, batch in enumerate(task.train_dl):
                memory.append([batch, task.id])
                if i >= batches_per_task:
                    break
        log.info(f"Loading the trained model")
        path = CKPT_DIR/f"{self.config.model_ckpt}.pt"
        self.model.load(path)
        log.info(f"Re-training using memory")
        examples_seen = 0
        wandb.watch(self.model, log="gradients", log_freq=self.config.eval_freq)
        for i, (batch, task.id) in enumerate(memory):
            loss, metric = self.model.step(batch, task.id) 
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            examples_seen += self.config.batch_size
            wandb.log({f"train/{task.id}/loss":loss.item()}, step=examples_seen)
            wandb.log(edit_keys(metric, f"train/{task.id}/"), step=examples_seen)
        log.info(f"Testing the re-trained model")
        log.info(f"Test examples: {[self.config.batch_size*2*len(task.test_dl) for task in self.tasks]}")
        task_metrics = {}
        for task in self.tasks:
            metrics = []
            for batch in task.test_dl:
                with torch.no_grad():
                    loss, metric = self.model.step(batch, task.id) 
                metrics.append(metric)
            task_metrics[task.id] = pd.DataFrame(metrics).mean().to_dict()
        report = edit_keys(flatten(task_metrics), "test/")
        for key, val in report.items():
            wandb.run.summary[key] = val
        log.info(f"Test Report:\n {report}")