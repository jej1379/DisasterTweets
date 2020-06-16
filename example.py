# https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04

from simpletransformers.classification import ClassificationModel
import os, random
import pandas as pd

args = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",

    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 128,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 8,
    "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,

    "logging_steps": 50,
    "save_steps": 2000,

    "overwrite_output_dir": False,
    "reprocess_input_data": False,
    "evaluate_during_training": False,

    "process_count": os.cpu_count() - 2 if os.cpu_count() > 2 else 1,
    "n_gpu": 1,
}


train_df = pd.read_csv('data/train.csv', usecols=[0,1,3,4])

size=train_df.shape[0]
train_df = pd.DataFrame({
    'id': train_df['id'],
    'label': train_df['target'],
    'alpha': train_df['keyword'],
    'text': train_df['text']
})

print(train_df.head())
train_idx=random.sample(range(size),  int(size*0.8))
dev_idx=list(set(range(size)).difference(set(train_idx)))

print("train: %d samples, dev: %d samples" %(len(train_idx), len(dev_idx)))
train_df.iloc[train_idx].to_csv('data/train.tsv', sep='\t', index=False, header=False)
train_df.iloc[dev_idx].to_csv('data/dev.tsv', sep='\t', index=False, header=False)

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base', num_labels=2, use_cuda=False)
model.train_model(train_df)
