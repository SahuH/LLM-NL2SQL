import numpy as np
import json
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoConfig
from transformers import Trainer
from transformers import TrainingArguments, EvalPrediction
from transformers import T5Tokenizer, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset


import random, warnings
warnings.filterwarnings("ignore")

config = AutoConfig.from_pretrained("t5-base",)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base",config=config,)
tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)


spider_dataset = load_dataset("spider")


class SpiderDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        db_id = self.data[idx]['db_id']
        schema = self.get_schema(db_id)

        input_text = f"translate English to SQL: {self.data[idx]['question']} <schema> {schema} </s>"
        target_text = self.data[idx]['query']
        encoding = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        target = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target['input_ids'].flatten()
        }

    def get_schema(self, db_id):
        schema_data = next(item for item in tables_data if item['db_id'] == db_id)
        schema = " ".join([f"Table: {table_name} Columns: {', '.join([col_name for _, col_name in schema_data['column_names'] if schema_data['table_names'][table_idx] == table_name])}" for table_idx, table_name in enumerate(schema_data['table_names'])])
        return schema

# Load tables.json
with open('tables.json', 'r') as f:
    tables_data = json.load(f)

train_dataset = SpiderDataset(spider_dataset['train'], tokenizer, max_length=128)
val_dataset = SpiderDataset(spider_dataset['validation'], tokenizer, max_length=128)



training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=1,
    learning_rate=1e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy='epoch',
    evaluation_strategy='epoch',
)

trainer = Trainer(
    model=model1,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

