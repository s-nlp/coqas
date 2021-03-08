import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss

from tqdm import tqdm_notebook, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multiprocessing import Pool, cpu_count
from tools import *
import convert_examples_to_features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_NAME = 'comparative'
MODEL_DIR = 'model/'
MAX_SEQ_LENGTH = 30
TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 20
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'
output_mode = OUTPUT_MODE
BERT_MODEL = "bert_comp.tar.gz"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
DATA_DIR = "data/"


def target_to_int(value):
    if value == 'BETTER':
        return 0
    elif value == 'WORSE':
        return 1
    elif value == 'NONE':
        return 2

def int_to_target(value):
    if value == 0:
        return 'BETTER'
    elif value == 1:
        return "WORSE"
    elif value == 2:
        return "NONE"

def return_prediction(data):
    df = pd.DataFrame()
    df['max'] = data
    df['max'] = df['max'].apply(int_to_target)
    return df

def get_bert_format(df):
    df_bert = pd.DataFrame({
        'id': range(len(df)),
        'label': [0] * df.shape[0],
        'alpha': ['a'] * df.shape[0],
        'text': df["object_a"] + ' [SEP] ' + df["object_b"] + ' [SEP] ' + df["sentence"]
    })
    return df_bert

def get_BERT_prediction(data):

    df_test_bert = get_bert_format(data)
    df_test_bert.to_csv(DATA_DIR + 'test.tsv', sep='\t', index=False, header=False)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR + 'vocab.txt', do_lower_case=False)
    processor = MultiClassificationProcessor()
    eval_examples = processor.get_test_examples(DATA_DIR)
    eval_examples_len = len(eval_examples)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}
    eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in
                                    eval_examples]
    process_count = cpu_count() - 1
    print(f'Preparing to convert {eval_examples_len} examples..')
    print(f'Spawning {process_count} processes..')
    with Pool(process_count) as p:
        eval_features = list(tqdm_notebook(
            p.imap(convert_examples_to_features.convert_example_to_feature, eval_examples_for_processing),
            total=eval_examples_len))
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    if OUTPUT_MODE == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)
    # Load pre-trained model (weights)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR + BERT_MODEL, num_labels=len(label_list))
    model.to(device)
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Prediction"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if OUTPUT_MODE == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if OUTPUT_MODE == "classification":
        preds = np.argmax(preds, axis=1)
    return return_prediction(preds)


# names = ["object_a", "object_b", "most_frequent_label", "sentence"]
# df_test = pd.read_csv("test_data.csv", usecols=names)
# preds = get_BERT_prediction(df_test)
# print(preds)


