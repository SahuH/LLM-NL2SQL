import os
import numpy as np
import pandas as pd
import re
import json
import argparse


def get_entities(schema_path):
    with open(schema_path, 'r', encoding='utf8') as f:
        table_datas = json.load(f)

    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(table_datas)):
        table = table_datas[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        tmp_col = []
        for cc in [x[1] for x in table['column_names_original']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        tabel_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table

    return tables

def correct_table_reference(db_dict, db_id, pred):
  db = db_dict[db_id]
  dict_1 = {}

  for info in db['column_names_original']:
    key = info[1].lower()
    value = db['table_names_original'][info[0]]
    if key in dict_1:
      old_value = dict_1[key]
    else:
      old_value = []
    old_value.append(value.lower())
    dict_1[key] = old_value

  table_vars = re.findall(r'(\w+) as (\w+)', pred)

  dict_2 = {}
  for key, val in table_vars:
    dict_2[key] = val

  columns = re.findall(r'select (.*?) from', pred)
  if columns != []:
    columns = columns[0]
    columns = columns.split(',')

    agg_fn_lst = ['count','sum','avg','min','max']
    for agg_fn in agg_fn_lst:
      columns = [term.replace(agg_fn+'(','') for term in columns]
      columns = [term.replace(')','') for term in columns]

    columns = [term.strip() for term in columns]
    if '*' in columns:
      columns.remove('*')
    # print(dict_1, dict_2)
    # print(columns)
    dict_3 = {}
    for col in columns:
      if '.' in col:
        [table_var, col_name] = col.split('.')[-2:]
        if col_name in dict_1:
          for table_name in dict_1[col_name]:
            if table_name in dict_2:
              # print(table_var, dict_2[table_name])
              pred = pred.replace(' '+table_var+'.', ' '+dict_2[table_name]+'.', 1)
              break

  return pred


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--schemas_path', dest='schemas_path', type=str, default='../spider_dataset/tables.json', help="the path to tables.json")
  parser.add_argument('--output_path', dest='output_path', type=str, default='../output/output.csv', help="the path to output.csv that containes gold, pred, db")
  args = parser.parse_args()

  db_dict = get_entities(args.schemas_path)     # Use 'col_set' from db_dict[<db_id>'] to match column names, 'table_names' from db_dict[<db_id>] to match table names
  output = pd.read_csv(args.output_path)

  row = output.iloc[0]
  preds_ec_trc = []
  for idx, row in output.iterrows():
    db_id = row['db']
    gold = row['gold']
    pred = row['pred_ec'].lower()
    print(db_id)
    print(pred)
    pred = correct_table_reference(db_dict, db_id, pred)
    print(pred)
    preds_ec_trc.append(pred)

  output['pred_ec_trc'] = preds_ec_trc
  output.to_csv(args.output_path, index=False)