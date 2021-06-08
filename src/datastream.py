import random
from functools import  partial
from datasets import load_dataset, concatenate_datasets


# TODO: DatastreamGenerator/LifelongStreamer converts a normal dataset to datastream
# TODO: Enforce datastream Features to have context, statement, label
class LifelongDatasetInterface:
    path: str 
    name: str
    
    def __init__(self, split, tokenizer, selected_indexes=None, cache_dir=None): 
        self.dataset = load_dataset(self.path, self.name, split=split, cache_dir=cache_dir)
        if selected_indexes and max(selected_indexes) <= self.dataset.num_rows:
            self.dataset = self.dataset.select(selected_indexes)
        self.dataset = self.dataset.map(partial(self.preprocess, tokenizer=tokenizer), 
                                        batched=True, 
                                        remove_columns=self.dataset.column_names)
    
    def preprocess(self, batch, tokenizer):
        pass

# TODO: Generalized mapper based on type of task

class BoolQ(LifelongDatasetInterface):
    path = "super_glue"
    name = "boolq"
    
    def preprocess(self, batch, tokenizer):
        contexts = batch["passage"]
        statements = batch["question"]
        features = tokenizer.batch_encode_plus(list(zip(contexts, statements)),
                                               padding='max_length',
                                               truncation="only_first")
        features["label"] = batch["label"]
        return features

class CB(LifelongDatasetInterface):
    path = "super_glue"
    name = "cb"
    task_descriptors = [". This implies ", ". This is "]

    def preprocess(self, batch, tokenizer):
        label2string = self.dataset.features['label'].int2str
        label_names = self.dataset.features['label'].names
        inputs, outputs = [], []
        for context, hypothesis, label  in zip(batch["premise"], 
                                               batch["hypothesis"],
                                               batch["label"]):
            desc = random.choice(self.task_descriptors)
            statement = " ".join([hypothesis, desc, label2string(label)])
            inputs.append([context, statement])
            outputs.append(1)
            for other_label in label_names:
                if other_label != label2string(label):
                    statement = " ".join([hypothesis, desc, other_label])
                    inputs.append([context, statement])
                    outputs.append(0)
        features = tokenizer.batch_encode_plus(inputs,
                                               padding='max_length',
                                               truncation="only_first")
        features["label"] = outputs
        return features

class COPA(LifelongDatasetInterface):
    path = "super_glue"
    name = "copa"
    
    def preprocess(self, batch, tokenizer):
        inputs, labels = [], []
        for row in zip(batch["premise"], 
                       batch["question"],
                       batch["choice1"],
                       batch["choice2"],
                       batch["label"]):
            context, question, choice0, choice1, correct_choice = row
            statement0 = " ".join([question, choice0])
            inputs.append([context, statement0])
            labels.append(1 if correct_choice==0 else 0)
            statement1 = " ".join([question, choice1])
            inputs.append([context, statement1])
            labels.append(1 if correct_choice==1 else 0)
        features = tokenizer.batch_encode_plus(inputs,
                                               padding='max_length',
                                               truncation="only_first")
        features["label"] = labels
        return features

class MultiRC(LifelongDatasetInterface):
    path = "super_glue"
    name = "multirc"
    
    def preprocess(self, batch, tokenizer):
        contexts = batch["paragraph"]
        statements = [" ".join([q, a]) for q, a in zip(batch["question"], batch["answer"])]
        features = tokenizer.batch_encode_plus(list(zip(contexts, statements)),
                                                padding='max_length',
                                                truncation="only_first")
        features["label"] = batch["label"]
        return features

class ReCoRD(LifelongDatasetInterface):
    path = "super_glue"
    name = "record"
    
    def preprocess(self, batch, tokenizer):
        inputs, labels = [], []
        for row in zip(batch["passage"], 
                       batch["query"],
                       batch["entities"],
                       batch["answers"]):
            context, query, choices, answers = row
            for choice in choices:
                statement = query.replace("@placeholder", choice)
                inputs.append([context, statement])
                labels.append(1 if choice in answers else 0)
        features = tokenizer.batch_encode_plus(inputs,
                                               padding='max_length',
                                               truncation="only_first")
        features["label"] = labels
        return features

class RTE(LifelongDatasetInterface):
    path = "super_glue"
    name = "rte"
    task_descriptors = ["This implies ", "This is "]
    
    def preprocess(self, batch, tokenizer):
        label2string = self.dataset.features['label'].int2str
        label_names = self.dataset.features['label'].names
        inputs, labels = [], []
        for context, hypothesis, label in zip(batch["premise"], 
                                              batch["hypothesis"],
                                              batch["label"]):
            desc = random.choice(self.task_descriptors)
            statement = " ".join([hypothesis, desc, label2string(label)])
            inputs.append([context, statement])
            labels.append(1)
            for other_label in label_names:
                if other_label != label2string(label):
                    statement = " ".join([hypothesis, desc, other_label])
                    inputs.append([context, statement])
                    labels.append(0)
        features = tokenizer.batch_encode_plus(inputs,
                                               padding='max_length',
                                               truncation="only_first")
        features["label"] = labels
        return features

class WIC(LifelongDatasetInterface):
    path = "super_glue"
    name = "wic"
    task_descriptors = [
        " is the polysemous word.", 
        " is used with the same sense."
    ]
    
    def preprocess(self, batch, tokenizer):
        contexts = [" ".join([sen1, sen2]) \
                    for sen1, sen2 in zip(batch["sentence1"], batch["sentence2"])]
        desc = random.choice(self.task_descriptors)
        statements = [" ".join([word, desc]) for word in batch["word"]]
        features = tokenizer.batch_encode_plus(list(zip(contexts, statements)),
                                                padding='max_length',
                                                truncation="only_first")
        features["label"] = batch["label"]
        return features

class WSC(LifelongDatasetInterface):
    path = "super_glue"
    name = "wsc"
    task_descriptors = [
        " refers to ", 
        " is ", 
        " is the pronoun of "
    ]
    
    def preprocess(self, batch, tokenizer):
        contexts = batch["text"]
        desc = random.choice(self.task_descriptors)
        statements = [" ".join([pronoun, desc, noun]) \
                        for noun, pronoun in zip(batch["span1_text"], batch["span2_text"])]
        features = tokenizer.batch_encode_plus(list(zip(contexts, statements)),
                                                padding='max_length',
                                                truncation="only_first")
        features["label"] = batch["label"]
        return features


def get_superglue_stream(tokenizer, examples_per_task, shuffle=False, cache_dir=None):
    dataset_classes = [BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WIC, WSC]
    if shuffle:
        random.shuffle(dataset_classes)
    indexes = range(examples_per_task)
    dataset_objs = [dataset_class("train", tokenizer, indexes, cache_dir) \
                    for dataset_class in dataset_classes]
    datastream = concatenate_datasets([obj.dataset for obj in dataset_objs])
    datastream_info = {
        obj.__class__.__name__: obj.dataset.num_rows 
        for obj in dataset_objs
    }
    return datastream, datastream_info
    

if __name__ == '__main__':
    ''' How to use? '''

    from transformers import BertTokenizer
    model_name = "bert-base-uncased"
    data_dir = "../../data"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 1. How to load a benchmark stream?
    # ---
    datastream, info = get_superglue_stream(tokenizer, 
                                            examples_per_task=1000,
                                            cache_dir=data_dir)
    print(f"Loaded the stream: {info}")

    # 2. How to customize your own stream?
    # ---
    multirc = MultiRC("train", tokenizer, cache_dir=data_dir)
    first_boolq = BoolQ("train", tokenizer, selected_indexes=range(0, 100), cache_dir=data_dir)
    wic = WIC("train", tokenizer, cache_dir=data_dir)
    wsc = WSC("train", tokenizer, cache_dir=data_dir)
    second_boolq = BoolQ("train", tokenizer, selected_indexes=range(100, 200), cache_dir=data_dir)

    custom_datastream = concatenate_datasets([multirc.dataset, 
                                              first_boolq.dataset,
                                              wic.dataset,
                                              wsc.dataset,
                                              second_boolq.dataset]) 

    for i in [0, 100, 500, 1000, -1000, -500, -100, -1]:
        input_id = tokenizer.decode(datastream[i]["input_ids"]).replace("[PAD]", "")
        label = datastream[i]["label"]
        print(input_id.rstrip())
        print(label)
        print()