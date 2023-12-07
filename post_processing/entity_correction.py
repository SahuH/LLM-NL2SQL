import math
import numpy as np
import pandas as pd
import argparse
import re
import json
import difflib

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

def get_matches(entity_lst, known_lst):
  match_dict = {}
  for entity in entity_lst:
    match_1 = difflib.get_close_matches(entity, known_lst, n=1, cutoff=0.3)
    if match_1 != []:
      match_dict[entity] = match_1[0]
    else:
      entity_lst.remove(entity)
  return entity_lst, match_dict

def replace_entity(pred, entity_lst, match_dict):
  for entity in entity_lst:
    pred = pred.replace(entity, match_dict[entity])
  return pred

# Table correction
def correct_tables(db_dict, db_id, pred):
  out = re.findall(r'from (.*?) (as|join|$)', pred)
  tables = [match[0] for match in out]

  out = re.findall(r'join (.*?) (where|groupby|having|$)', pred)
  table_2 =  [match[0] for match in out]
  tables = tables + table_2

  tables, match_dict = get_matches(tables, db_dict[db_id]['table_names_original'])
  pred = replace_entity(pred, tables, match_dict)

  return pred

# Column correction
def correct_columns(db_dict, db_id, pred):
  columns = re.findall(r'select (.*?) from', pred)
  if columns != []:
    columns = columns[0]
    columns = columns.split(',')

    agg_fn_lst = ['count','sum','avg','min','max']
    for agg_fn in agg_fn_lst:
      columns = [term.replace(agg_fn+'(','') for term in columns]
      columns = [term.replace(')','') for term in columns]

    columns = [re.sub(r'.*\.', '', term) for term in columns]

    columns = [term.strip() for term in columns]
    if '*' in columns:
      columns.remove('*')
    # print(columns)
    columns, match_dict = get_matches(columns, db_dict[db_id]['col_set'])
    # print(columns)
    pred = replace_entity(pred, columns, match_dict)

  return pred


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--schemas_path', dest='schemas_path', type=str, default='../spider_dataset/tables.json', help="the path to tables.json")
  parser.add_argument('--output_path', dest='output_path', type=str, default='../output/output.csv', help="the path to output.csv that containes gold, pred, db")
  args = parser.parse_args()

  db_dict = get_entities(args.schemas_path)     # Use 'col_set' from db_dict[<db_id>'] to match column names, 'table_names' from db_dict[<db_id>] to match table names
  output = pd.read_csv(args.output_path)

  preds_ec = []
  for idx, row in output.iterrows():
    db_id = row['db']
    gold = row['gold']
    pred = row['pred_picard'].lower()
    print(db_id)
    print(pred)
    try:
      pred = correct_tables(db_dict, db_id, pred)
      pred = correct_columns(db_dict, db_id, pred)
    except:
      pass
    print(pred)
    preds_ec.append(pred)

  output['pred_picard_ec'] = preds_ec
  output.to_csv(args.output_path, index=False)