from transformers import AutoModelWithLMHead, AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
import nltk
import json

import random, warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelWithLMHead.from_pretrained("t5-base")

def get_sql(query):
    
    input_text = "translate English to SQL: %s </s>" % query
    
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])

    return tokenizer.decode(output[0])

def get_sql_with_schema(query, schema):
    # Concatenate the schema information with the input query text, separated by a special token, such as `<schema>`
    input_text = f"translate English to SQL: {query} <schema> {schema} </s>"

    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'], 
                             attention_mask=features['attention_mask'])

    decoded_output = tokenizer.decode(output[0])

    # Remove the <pad> token from the output
    return decoded_output.replace('<pad>', '').strip()


spider_dataset = load_dataset('spider')  # Using spider dataset from huggingface datasets library (and not the one stored in local)


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
            'labels': target['input_ids'].flatten(),
            'data_item': self.data[idx] 
        }

    def get_schema(self, db_id):
        schema_data = next(item for item in tables_data if item['db_id'] == db_id)
        schema = " ".join([f"Table: {table_name} Columns: {', '.join([col_name for _, col_name in schema_data['column_names'] if schema_data['table_names'][table_idx] == table_name])}" for table_idx, table_name in enumerate(schema_data['table_names'])])
        return schema

# Load tables.json
with open('./spider_dataset/tables.json', 'r') as f:
    tables_data = json.load(f)

train_dataset = SpiderDataset(spider_dataset['train'], tokenizer, max_length=128)
val_dataset = SpiderDataset(spider_dataset['validation'], tokenizer, max_length=128)


gold_file = open('gold.txt', 'w')
pred_file = open('pred.txt', 'w')


count = 0
N = len(val_dataset)
for idx in range(len(val_dataset)):
    item = val_dataset[idx]
    data_item = item['data_item']
    print(f'{count + 1}/{N}')
    print(f"Text: {data_item['question']}")

    # Get schema information
    db_id = data_item['db_id']
    schema = val_dataset.get_schema(db_id)

    pred = get_sql_with_schema(data_item['question'], schema)
    gold = data_item['query']

    gold_file.write(gold + '\t' + db_id + '\n')
    pred_file.write(pred + '\n')
    
    print(f"Pred SQL: {pred}")
    print(f"True SQL: {gold}\n")

    count += 1
  
gold_file.close()
pred_file.close()
