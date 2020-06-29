# -*- coding:utf-8 -*-

import torch
import pickle
import utils
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.nn import CrossEntropyLoss

from tqdm import tqdm, trange
import os, datetime
from pytorch_pretrained_bert import BertTokenizer
import modeling
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool, cpu_count
import convert_examples_to_features

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The input data dir. Should contain the .tsv files (or other data files) for the task.
DATA_DIR = "data/"

# Bert pre-trained model selected in the list: bert-base-uncased,
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-base-uncased'
# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'DisasterTweets'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = f'outputs/'

# The directory where the evaluation reports will be written to.
REPORTS_DIR = f'reports/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = 'cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
# avg(seq_len) ~ 31, max(seq_len) = 82 (w/ BertTokenizer)
MAX_SEQ_LENGTH = 50
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 5e-4
NUM_TRAIN_EPOCHS = 2
RANDOM_SEED = 42
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'
CONFIG_NAME = 'config.json'

output_mode = OUTPUT_MODE
cache_dir = CACHE_DIR

processor = utils.BinaryClassificationProcessor()
label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

process_count = cpu_count() - 1

def get_dataloader(typ='test', BATCH_SIZE=TRAIN_BATCH_SIZE, shuffle=False):
    examples = processor.get_test_examples(DATA_DIR)
    examples_len = len(examples)

    label_map = {label: i for i, label in enumerate(label_list)}
    if os.path.exists(DATA_DIR + "%s_features.pkl" %typ):
        features = pickle.load(open(DATA_DIR + "%s_features.pkl" %typ, "rb"))
    else:
        examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples]
        with Pool(process_count) as p:
            features = list(tqdm(p.imap(convert_examples_to_features.convert_example_to_feature, examples_for_processing),total=examples_len))

        with open(DATA_DIR + "%s_features.pkl" %typ, "wb") as f:
            pickle.dump(features, f)

    data = TensorDataset(
        torch.tensor([int(f.ids) for f in features], dtype=torch.int),
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.input_mask for f in features], dtype=torch.long),
        torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        torch.tensor([0 for f in features], dtype=torch.long))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)
    return dataloader

model = modeling.BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
model.load_state_dict(torch.load('./outputs/pytorch_model_90.6250.bin'))
model.to(device)

if __name__ ==  '__main__':
    print(f'Preparing to convert test examples..')
    print(f'Spawning {process_count} processes..')
    test_dataloader = get_dataloader('test', EVAL_BATCH_SIZE)

    # best_model = modeling.BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
    # best_model.load_state_dict(torch.load(PATH))

    model.eval()
    step, saved = 0, 0
    save_file=open('result.csv')
    for batch in tqdm(test_dataloader, desc="Iteration"):
        batch = tuple(t.to(device) for t in batch)
        ids, input_ids, input_mask, segment_ids, label_ids = batch

        logits, _ = model(input_ids, segment_ids, input_mask, labels=label_ids)

        prediction = torch.max(logits.view(-1, num_labels), 1).indices
        for id, p in zip(ids, prediction):
            save_file.write('%d,%d\n' %(id, p))
            saved+=1
        step += 1
    print('Test Set Result w/ %d samples' %saved)