import json
import sys

def process_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    with open('gold.txt', 'w') as gold_file, open('pred.txt', 'w') as pred_file:
        for item in data:
            gold = item['query']
            prediction_full = item['prediction']
            pred = prediction_full.split('|')[1].strip() if '|' in prediction_full else prediction_full.strip()

            gold_file.write(gold + '\t' + item['db_id'] + '\n')
            pred_file.write(pred + '\n')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python output.py <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    process_data(json_file_path)

