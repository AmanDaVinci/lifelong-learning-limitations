import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

def avg(dictionary):
    ''' Compute average of all dictionary values '''
    return np.mean([val for val in dictionary.values()])

def normalize(tensor):
    ''' Min-max normalize a tensor '''
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def edit_keys(dictionary, prefix="", postfix=""):
    '''Edit keys of a dict with a prefix and postfix'''
    return {f"{prefix}{key}{postfix}":val for key,val in dictionary.items()}

def flatten(dictionary, sep="/"):
    '''Flattens a nested dict (2 levels only) by joining keys'''
    return {f"{key}{sep}{inner_key}":val for key,inner_dict in dictionary.items()\
                                         for inner_key,val in inner_dict.items()}
