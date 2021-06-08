import wandb
import random
import torch
import logging
from pathlib import Path
from argparse import ArgumentParser

from transformers import (
    AdamW, 
    AutoModel, 
    AutoTokenizer, 
    GlueDataset, 
    GlueDataTrainingArguments, 
    default_data_collator
)

from src.trainer import Trainer
from src.multitask_trainer import MultiTaskTrainer
from src.test_time_trainer import TestTimeTrainer

def main():
    # TODO: config system
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='lifelong')
    parser.add_argument('--model_ckpt', type=str)
    parser.add_argument('--memory_size', type=int, default=5)
    parser.add_argument('--base_model', type=str, default='bert-base-uncased')
    parser.add_argument('--head_model', type=str, default='multi')
    parser.add_argument('--task_stream', type=str, default='domainshift')
    parser.add_argument('--mask', type=str, default='None')
    parser.add_argument('--mask_cutoff', type=float, default=0.0)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=400)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--mask_data_size', type=int, default=500)
    parser.add_argument('--max_eval_size', type=int, default=500)
    parser.add_argument('--max_train_size', type=int, default=10000)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--grad_sim_freq', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    with wandb.init(project="lifelong-learning-limitations", config=args):
        if args.mode == "lifelong":
            trainer = Trainer(args)
        elif args.mode == "multitask":
            trainer = MultiTaskTrainer(args)
        elif args.mode == "test_time_trainer":
            trainer = TestTimeTrainer(args)
        else:
            raise Exception("Undefined training mode")
        trainer.run()


if __name__ == "__main__":
    main()
