import json
import random
import nltk
import numpy as np
from transformers import T5Tokenizer
from torch.utils.data import Dataset
from transformers import AutoConfig
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset
from adapters import AutoAdapterModel
from transformers import TrainingArguments, EvalPrediction
from adapters import AdapterTrainer
from adapters import AutoAdapterModel

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
with open('/content/drive/MyDrive/spider/tables.json', 'r') as f:
    tables_data = json.load(f)

train_dataset = SpiderDataset(spider_dataset['train'], tokenizer, max_length=128)
val_dataset = SpiderDataset(spider_dataset['validation'], tokenizer, max_length=128)


config = AutoConfig.from_pretrained(
    "t5-base",
)
model = AutoAdapterModel.from_pretrained(
    model='t5-base',
    config=config,
)

# Add a new adapter
model.add_adapter("nlp")
# Activate the adapter
model.train_adapter("nlp")


training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    remove_unused_columns=False,
)


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
