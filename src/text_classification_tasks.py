import pandas as pd
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import datasets
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from src.utils import DataCollator


def initialize(task_stream: str) -> list:
    # GLUE
    cola = TextClassificationTask(path="glue", 
                                  config_name="cola", 
                                  text_columns=["sentence"])
    mnli = TextClassificationTask(path="glue", 
                                  config_name="mnli", 
                                  text_columns=["premise"],
                                  secondtext_columns=["hypothesis"], 
                                  num_labels=3,
                                  eval_key="validation_matched")
    mrpc = TextClassificationTask(path="glue", 
                                  config_name="mrpc", 
                                  text_columns=["sentence1"],
                                  secondtext_columns=["sentence2"])
    qnli = TextClassificationTask(path="glue", 
                                  config_name="qnli", 
                                  text_columns=["question"],
                                  secondtext_columns=["sentence"])
    qqp = TextClassificationTask(path="glue", 
                                 config_name="qqp", 
                                 text_columns=["question1"],
                                 secondtext_columns=["question2"])
    rte = TextClassificationTask(path="glue", 
                                 config_name="rte", 
                                 text_columns=["sentence1"],
                                 secondtext_columns=["sentence2"])
    sst2 = TextClassificationTask(path="glue", 
                                  config_name="sst2", 
                                  text_columns=["sentence"])
    stsb = TextClassificationTask(path="glue", 
                                  config_name="stsb", 
                                  text_columns=["sentence1"],
                                  secondtext_columns=["sentence2"], 
                                  num_labels=1)
    wnli = TextClassificationTask(path="glue", 
                                  config_name="wnli", 
                                  text_columns=["sentence1"],
                                  secondtext_columns=["sentence2"])

    # Domain shift
    yrf = TextClassificationTask(path="yelp_review_full", 
                                 text_columns=["text"], 
                                 num_labels=5, 
                                 metric_name="accuracy", 
                                 eval_key="test")
    agn = TextClassificationTask(path="ag_news", 
                                 text_columns=["text"], 
                                 num_labels=4, 
                                 metric_name="accuracy", 
                                 eval_key="test")
    dbp = TextClassificationTask(path="dbpedia_14", 
                                 text_columns=["title", "content"],
                                 num_labels=14, 
                                 metric_name="accuracy", 
                                 eval_key="test")
    arm = TextClassificationTask(path="amazon_reviews_multi", 
                                 config_name="en", 
                                 text_columns=["review_title", "review_body"],
                                 label_column="stars",
                                 num_labels=5, 
                                 metric_name="accuracy")
    yat = TextClassificationTask(path="yahoo_answers_topics", 
                                 text_columns=["question_title", "question_content", "best_answer"],
                                 label_column="topic",
                                 num_labels=10, 
                                 metric_name="accuracy", 
                                 eval_key="test")
    if task_stream=="glue":
        tasks = [cola, mnli, mrpc, sst2, stsb, qqp, qnli, rte, wnli]
    elif task_stream=="small_glue":
        tasks = [cola, mrpc, stsb, rte, wnli]
    elif task_stream=="domainshift":
        tasks = [yrf, agn, dbp, arm, yat]
    elif task_stream=="domainshift_2":
        tasks = [dbp, yat, agn, arm, yrf]
    elif task_stream=="domainshift_3":
        tasks = [yrf, yat, arm, dbp, agn]
    elif task_stream=="domainshift_4":
        tasks = [agn, yrf, arm, yat, dbp]
    else:
        raise Exception("Task stream not defined")
    return tasks


@dataclass
class TextClassificationTask:
    path: str
    text_columns: List[str]
    label_column: str = "label"
    num_labels: int = 2
    metric_name: str = ""
    config_name: Optional[str] = None
    train_key: str = "train"
    eval_key: str = "validation"
    secondtext_columns: List[str] = field(default_factory=list)
        
    def preprocess_(self, 
                   tokenizer: PreTrainedTokenizerBase, 
                   data_dir: Path,
                   batch_size: int,
                   max_seq_len: int,
                   mask_data_size: int, 
                   max_train_size: Optional[int] = None,   
                   max_eval_size: Optional[int] = None):
        '''Loads and preprocesses the data to prepare dataloaders in-place'''
        self.id = self.path+"_"+self.config_name if self.config_name else self.path
        if self.metric_name:
            self.metric = load_metric(self.metric_name, keep_in_memory=True)
        else:
            self.metric = load_metric(self.path, self.config_name, keep_in_memory=True)
            
        datasets = load_dataset(self.path, self.config_name, cache_dir=data_dir)
        # for classification tasks only
        if self.num_labels > 1:
            label_list = datasets["train"].unique(self.label_column)
            self.label_map = {name: idx for idx, name in enumerate(sorted(label_list))}
        else:
            self.label_map = None
        preprocess_fn = partial(preprocess,
                                tokenizer=tokenizer,
                                max_len=max_seq_len,
                                text_columns=self.text_columns,
                                label_column=self.label_column,
                                label_map=self.label_map, 
                                secondtext_columns=self.secondtext_columns)
        datasets = datasets.map(preprocess_fn, 
                                batched=True, 
                                remove_columns=self.text_columns+self.secondtext_columns)
        if self.label_column!="label":
            datasets.rename_column_(self.label_column, "label")
        train_dataset = datasets[self.train_key]
        eval_dataset = datasets[self.eval_key]
        test_dataset = datasets[self.eval_key]
        mask_dataset = trim(train_dataset, mask_data_size)
        if max_train_size:
            train_dataset = trim(train_dataset, max_train_size)
        if max_eval_size:
            eval_dataset = trim(eval_dataset, max_eval_size)
        required_columns = ['input_ids', 'token_type_ids', 'attention_mask', 'label']
        train_dataset.set_format(type='torch', columns=required_columns)
        eval_dataset.set_format(type='torch', columns=required_columns)
        test_dataset.set_format(type='torch', columns=required_columns)
        mask_dataset.set_format(type='torch', columns=required_columns)
        
        collator = DataCollator(tokenizer)
        train_sampler, eval_sampler = RandomSampler(train_dataset), SequentialSampler(eval_dataset)
        mask_sampler, test_sampler = RandomSampler(mask_dataset), SequentialSampler(test_dataset)
        self.train_dl = DataLoader(train_dataset, batch_size, sampler=train_sampler, collate_fn=collator)
        self.eval_dl = DataLoader(eval_dataset, batch_size, sampler=eval_sampler, collate_fn=collator)
        self.test_dl = DataLoader(test_dataset, batch_size*2, sampler=test_sampler, collate_fn=collator)
        self.mask_dl = DataLoader(mask_dataset, batch_size, sampler=mask_sampler, collate_fn=collator)
        

def preprocess(batch: Dict[str, List], 
               tokenizer: PreTrainedTokenizerBase, 
               max_len: int, 
               text_columns: List[str], 
               label_column: str, 
               label_map: Dict[Any, int] = None, 
               secondtext_columns: List[str] = None) -> Dict[str, list]:
    df = pd.DataFrame(batch)
    if label_map:
        labels = df[label_column].map(label_map).tolist()
    else:
        labels = df[label_column].tolist()
    df["text"] = df[text_columns].agg(" ".join, axis=1)
    if secondtext_columns:
        df["secondtext"] = df[secondtext_columns].agg(" ".join, axis=1)
    batch=df.to_dict(orient="list")
    if secondtext_columns: 
        batch = tokenizer(batch["text"], batch["secondtext"], 
                          truncation="longest_first", max_length=max_len)
    else:
        batch = tokenizer(batch["text"], 
                          truncation="longest_first", max_length=max_len)
    batch[label_column] = labels
    return batch

def trim(dataset: datasets.Dataset, size: int) -> datasets.Dataset:
    size = size if dataset.num_rows>size else (dataset.num_rows-1)
    slim_dataset = dataset.train_test_split(test_size=size)["test"]
    return slim_dataset