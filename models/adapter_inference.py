import json
import random
import nltk
from transformers import T5Tokenizer
from torch.utils.data import Dataset
from transformers import AutoConfig
from adapters import AutoAdapterModel
from datasets import load_dataset, load_metric
nltk.download('punkt')


spider_dataset = load_dataset("spider")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model_path = 't5finetuned'
adapter_path = 'adapternlp2sql'


config = AutoConfig.from_pretrained(
    "t5-base",
)
model = AutoAdapterModel.from_pretrained(
    model_path,config=config
)

model.load_adapter(adapter_path,model_name='nlp')
model.set_active_adapters('nlp')

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
      


def get_sql_with_schema(query, schema):
    # Concatenate the schema information with the input query text, separated by a special token, such as `<schema>`
    input_text = f"translate English to SQL: {query} <schema> {schema} </s>"

    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'], 
                             attention_mask=features['attention_mask'])

    decoded_output = tokenizer.decode(output[0])

    # Remove the <pad> token from the output
    return decoded_output.replace('<pad>', '').strip()


gold_file = open('gold.txt', 'w')
pred_file = open('pred.txt', 'w')

count = 0
for idx in range(len(val_dataset)):
    item = val_dataset[idx]
    # Access the original data using idx to get question and db_id
    question = val_dataset.data[idx]['question']
    db_id = val_dataset.data[idx]['db_id']
    gold_query = val_dataset.data[idx]['query']  # Assuming 'query' is the correct key for the SQL query

    print(f'{idx + 1}/{len(val_dataset)}')
    print(f"Text: {question}")

    # Get schema information
    schema = val_dataset.get_schema(db_id)

    pred_query = get_sql_with_schema(question, schema)

    gold_file.write(gold_query + '\t' + db_id + '\n')
    pred_file.write(pred_query + '\n')
    
    print(f"Pred SQL: {pred_query}")
    print(f"True SQL: {gold_query}\n")
    count += 1
gold_file.close()
pred_file.close()
# Load tables.json
with open('tables.json', 'r') as f:
    tables_data = json.load(f)



train_dataset = SpiderDataset(spider_dataset['train'], tokenizer, max_length=128)
val_dataset = SpiderDataset(spider_dataset['validation'], tokenizer, max_length=128)
