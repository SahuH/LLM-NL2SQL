
import numpy as np
import pandas as pd

gold_file_path = '../output/gold.txt'
pred_file_path = '../output/pred_picard.txt'

with open(gold_file_path, 'r') as gold_file, open(pred_file_path, 'r') as pred_file:
    gold_lst = gold_file.readlines()
    pred_lst = pred_file.readlines()

gold_lst_1 = []
db_lst = []
for gold in gold_lst:
    gold, db = gold.split('\t')
    db  = db.strip('\n')
    gold_lst_1.append(gold)
    db_lst.append(db)

pred_lst_1 = []
for pred in pred_lst:
    pred = pred.strip('\n')
    pred_lst_1.append(pred)


df = pd.DataFrame({'gold': gold_lst_1, 'db': db_lst, 'pred_picard': pred_lst_1})
df.to_csv('../output/output.csv', index=False)