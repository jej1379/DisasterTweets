# -*- coding:utf-8 -*-

import torch
import pickle, glob
import utils
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from tqdm import tqdm, trange
import os, datetime
from pytorch_pretrained_bert import BertTokenizer
import modeling
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool, cpu_count
import convert_examples_to_features

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 10
WARMUP_PROPORTION = 0.05
RANDOM_SEED = 42
OUTPUT_MODE = 'classification'
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model_%d_%.2f.bin'

output_mode = OUTPUT_MODE
cache_dir = CACHE_DIR

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

processor = utils.BinaryClassificationProcessor()
label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)

train_examples_len = len(processor.get_train_examples(DATA_DIR))
num_train_optimization_steps = int(train_examples_len / TRAIN_BATCH_SIZE) * NUM_TRAIN_EPOCHS


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
process_count = int(.6*cpu_count())

def get_dataloader(typ='train', BATCH_SIZE=TRAIN_BATCH_SIZE, shuffle=False):
    label_map = {label: i for i, label in enumerate(label_list)}

    if os.path.exists(DATA_DIR + "%s_features.pkl" %typ):
        features = pickle.load(open(DATA_DIR + "%s_features.pkl" %typ, "rb"))
    else:
        if typ == 'train': examples = processor.get_train_examples(DATA_DIR)
        elif typ == 'dev': examples = processor.get_dev_examples(DATA_DIR)
        examples_len = len(examples)
        examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in examples]
        with Pool(process_count) as p:
            features = list(tqdm(p.imap(convert_examples_to_features.convert_example_to_feature, examples_for_processing),
                                     total=examples_len))

        with open(DATA_DIR + "%s_features.pkl" %typ, "wb") as f:
            pickle.dump(features, f)

    data = TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                             torch.tensor([f.input_mask for f in features], dtype=torch.long),
                             torch.tensor([f.segment_ids for f in features], dtype=torch.long),
                             torch.tensor([f.label_id for f in features], dtype=torch.long))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)
    return dataloader

def eval(best_model, dev_dataloader):
    best_model.eval()
    dev_loss, dev_acc, dev_step = 0, .0, 0
    for batch in dev_dataloader: #tqdm(dev_dataloader, desc="Iteration"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        logits, loss = best_model(input_ids, segment_ids, input_mask, labels=label_ids)

        acc = (torch.max(logits.view(-1, num_labels), 1).indices == label_ids).sum().item() / float(EVAL_BATCH_SIZE)
        #print("\neval steps = %d, Loss = %.4f, Acc = %.2f" % (dev_step, loss, acc))

        dev_acc += acc
        dev_loss += loss.item()
        dev_step += 1
    print('\nDev Set Result w/ ')
    avg_acc, avg_loss = 100 * float(dev_acc)/dev_step, float(dev_loss)/dev_step
    print('Avg Acc = %.4f, Total Step = %d, Avg Loss = %.4f' %(avg_acc, dev_step, avg_loss))
    return avg_acc, avg_loss

if __name__ ==  '__main__':
    print(f'Preparing to convert {train_examples_len} examples..')
    print(f'Spawning {process_count} processes..')

    # Load pre-trained model (weights)
    model = modeling.BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=LEARNING_RATE,
                         #warmup=WARMUP_PROPORTION,
                         t_total=num_train_optimization_steps)
    global_step, nb_tr_steps = 0, 0
    tr_loss = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d, # label = %d" %(train_examples_len, num_labels))
    logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    dev_dataloader = get_dataloader('dev', EVAL_BATCH_SIZE)
    train_dataloader = get_dataloader('train', TRAIN_BATCH_SIZE, shuffle=True)
    # tensorboard --logdir=reports
    writer = SummaryWriter('./reports/train_{0}-{1}'.format(TASK_NAME, datetime.datetime.now().strftime("%Y%m%d_%H%M")))

    best = 80.
    for epoch in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps, eval_acc = 0, 0, .0
        model.train()
        for batch in train_dataloader: #tqdm(train_dataloader, desc="Iteration"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits, loss = model(input_ids, segment_ids, input_mask, labels=label_ids)
            loss.backward()
            acc = 100 * (torch.max(logits.view(-1, num_labels), 1).indices == label_ids).sum().item() / float(
                TRAIN_BATCH_SIZE)

            if global_step % 30 == 0:
                eval_acc, eval_loss = eval(model, dev_dataloader)
                best = eval_acc if eval_acc > best else best
                if eval_acc >= best:
                    # If we save using the predefined names, we can load using `from_pretrained`
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    torch.save(model_to_save.state_dict(), os.path.join(OUTPUT_DIR, WEIGHTS_NAME % (epoch, best)))
                    model_to_save.config.to_json_string()
                    tokenizer.save_vocabulary(OUTPUT_DIR)
                    print("\n", WEIGHTS_NAME % (epoch, best), "SAVED!")

                print("Train steps = %d, Loss = %.4f, Acc = %.2f(Best Eval = %.2f)\n" % (global_step, loss, acc, best))
                writer.add_scalar('Train/Loss', loss, global_step)
                writer.add_scalar('Train/Acc', acc, global_step)

                writer.add_scalar('Val/Loss', eval_loss, global_step)
                writer.add_scalar('Val/Acc', eval_acc, global_step)
                writer.flush()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
