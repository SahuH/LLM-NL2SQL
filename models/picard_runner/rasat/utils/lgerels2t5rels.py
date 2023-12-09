from json.tool import main
import pickle
from tkinter.tix import MAIN
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer
import itertools
import sys
sys.path.append('/home/jytang/NLP/text2sql/PLM/treetext2sql/text2sql-lgesql-main/')# need to change


tokenizer = AutoTokenizer.from_pretrained('t5-small')
MAX_RELATIVE_DIST = 2
RELATIONS = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    ['table-table-identity', 'table-table-fk', 'table-table-fkr', 'table-table-fkb'] + \
    ['column-column-identity', 'column-column-sametable', 'column-column-fk', 'column-column-fkr'] + \
    ['table-column-pk', 'column-table-pk', 'table-column-has', 'column-table-has'] + \
    ['question-column-exactmatch', 'question-column-partialmatch', 'question-column-nomatch', 'question-column-valuematch',
    'column-question-exactmatch', 'column-question-partialmatch', 'column-question-nomatch', 'column-question-valuematch'] + \
    ['question-table-exactmatch', 'question-table-partialmatch', 'question-table-nomatch',
    'table-question-exactmatch', 'table-question-partialmatch', 'table-question-nomatch'] + \
    ['question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic'] + \
    ['*-*-identity', '*-question-generic', 'question-*-generic', '*-table-generic', 'table-*-generic', '*-column-generic', 'column-*-generic', 'has-dbcontent', 'has-dbcontent-r']
relation2id_dict = dict(zip(RELATIONS, [i for i in range(1, len(RELATIONS)+1)]))

def quote_normalization(question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks = [], ['“', '”', '``', "''", "‘‘", "’’","'", '"', '`', '‘', '’']
    for idx, tok in enumerate(question):
        for mark in quotation_marks:
            tok = tok.replace(mark, "\"")
        new_question.append(tok)
    return new_question


def mul_mul_match(t5_toks_list,  question_toks_list):
    """"match two list of question toks"""
    t5_index = [i for i in range(1, len(t5_toks_list)+1)]
    question_index = [i for i in range(1, len(question_toks_list)+1)]
    index_pair = list(itertools.product(t5_index, question_index))
    for i, j in index_pair:
        t5_toks = "".join(t5_toks_list[:i])
        question_toks = "".join(question_toks_list[:j])
        if t5_toks == question_toks:
            return i, j
    return -1,-1

def match_question(t5_item, lge_item,  mode = "train"):
    flag = True
    question_lgeid2t5id = {}
    question_t5id2lgeid = {}
    lge_r_question_toks = lge_item['ori_toks']
    t5_toks_ids = t5_item
    t5_toks = []
    for id in t5_toks_ids:
        w = tokenizer.decode(id).replace("</s>", "")
        t5_toks.append(w)
    
    if mode == "dev" and "<unk>" in t5_toks:
        unk_idx = t5_toks.index("<unk>")
        t5_toks[unk_idx] = "\""
        pair_idx = t5_toks.index('\'')
        t5_toks[pair_idx] = "\""
        pair_idx = t5_toks.index('\'')
        t5_toks[pair_idx] = ""
    sep = t5_toks.index("|")
    t5_toks = t5_toks[:sep]
    t5_toks = quote_normalization([subword for subword in t5_toks])
    lge_r_question_toks = quote_normalization(lge_r_question_toks)
    lge_r_question_toks = [tokenizer.decode(tokenizer.encode(r_question_toks)[:-1]) for r_question_toks in lge_r_question_toks]
    t5_toks_ids = t5_item
    start = 0
    toks_idx = 0
    while toks_idx < len(lge_r_question_toks) and start < len(t5_toks):
        append_t5_idx, append_q_idx = mul_mul_match(t5_toks[start:],  lge_r_question_toks[toks_idx:])
        match_t5_id_list = [i for i in range(start,start+append_t5_idx)]
        match_q_id_list = [i for i in range(toks_idx,toks_idx+append_q_idx)]
        for q_idx in match_q_id_list:
            question_lgeid2t5id[q_idx] = match_t5_id_list
        for t5_idx in match_t5_id_list:
            question_t5id2lgeid[t5_idx] = match_q_id_list
        if append_t5_idx == -1 and append_q_idx == -1:
            flag = False
            break
        else:
            start += append_t5_idx
            toks_idx += append_q_idx
    res = {}
    res["question_lgeid2t5id"] = question_lgeid2t5id
    res["question_t5id2lgeid"] = question_t5id2lgeid
    return res, flag

def find_all_sep_index_from_list(item_list, sep_item):
    start = 0
    sep_list = []
    while start < len(item_list):
        try:
            index = item_list.index(sep_item, start)
            start = index+1
            sep_list.append(index)
        except:
            break
    sep_list.append(len(item_list))
    return sep_list

def find_all_sep_pair_from_list(item_list, sep_item_1, sep_item_2):
    start = 0
    sep_list = []
    while start < len(item_list):
        try:
            index_1 = item_list.index(sep_item_1, start)
            start = index_1+1
            index_2 = item_list.index(sep_item_2, start)
            start = index_2+1
            sep_list.append((index_1, index_2))
        except:
            break
    sep_list.append(len(item_list))
    return sep_list

def match_table_and_column(t5_item, lge_item, table_lgesql):
    column_lgeid2t5id = {}
    table_lgeid2t5id = {}
    dbcontent_lgeid2dbt5id = {}
    t5_toks_ids = t5_item
    
    t5_toks = []
    for id in t5_toks_ids:
        w = tokenizer.decode(id).replace("</s>", "")
        t5_toks.append(w)
        t5_toks = ([subword for subword in t5_toks])
    
    sep_index_list = find_all_sep_index_from_list(t5_toks, "|")
    
    db_name = "".join([w for w in t5_toks[sep_index_list[0]+1:sep_index_list[1]]])
    lge_item["db_name"] = db_name
    column_lgeid2t5id[0] = [i for i in range(sep_index_list[0]+1, sep_index_list[1])]
    
    lge_table = table_lgesql[db_name]['table_names']
    lge_table = [i.replace(" ", "_") for i in lge_table]
    lge_column = table_lgesql[db_name]['column_names']
    lge_column = [i[1].replace(" ", "_") for i in lge_column]
    
    lge_table_ori = [item.lower() for item in table_lgesql[db_name]['table_names_original']]
    lge_column_ori = [item[1].replace(" ", "").lower() for item in table_lgesql[db_name]['column_names_original']]
    
    flag = True
    for idx in range(1,len(sep_index_list)-1):
        item = [w for w in t5_toks[sep_index_list[idx]+1:sep_index_list[idx+1]]]
        table_bias = sep_index_list[idx]+1
        try:
            tb_col_sep_index = item.index(":")
        except:
            tb_col_sep_index = len(item)
        table = "".join([w for w in t5_toks[table_bias: table_bias+tb_col_sep_index]])
        try:
            lge_tb_idx = lge_table.index(table)
        except:
            try:
                lge_tb_idx = lge_table_ori.index(table)
            except:
                if tb_col_sep_index != len(item):
                    flag = False
                continue
        
        table_lgeid2t5id[lge_tb_idx] = [i for i in range(table_bias, table_bias+tb_col_sep_index)]
        
        db_content_bracket = find_all_sep_pair_from_list(item, "[", "]")[:-1]
        db_change_index_list = []
        if len(db_content_bracket)>0:
            for pair in db_content_bracket:
                pair_i, pair_j = pair
                for index in range(pair_i, pair_j):
                    if item[index] == ",":
                        item[index] = "~"
                        db_change_index_list.append(index)
        
        column_sep_index_list = find_all_sep_index_from_list(item, ",")
        for index in db_change_index_list:
            item[index] = ","
        column_sep_index_list.insert(0, tb_col_sep_index)
        column_bias = table_bias 
        for col_idx in range(1, len(column_sep_index_list)):
            col_lidx = column_bias+column_sep_index_list[col_idx-1]+1
            col_ridx = column_bias+column_sep_index_list[col_idx]
            col = t5_toks[col_lidx: col_ridx]
            db_content_lbracket = find_all_sep_index_from_list(col, "[")[:-1]
            db_content_rbracket = find_all_sep_index_from_list(col, "]")
            db_t5toks_list = []
            if len(db_content_lbracket) > 0:
                db_content_lbracket = db_content_lbracket[0]
                db_content_rbracket = db_content_rbracket[0]
                if col[db_content_rbracket-1] != "":
                    continue
                db_content = col[db_content_lbracket+1:db_content_rbracket]
                db_sep_index_list = find_all_sep_index_from_list(db_content, ";")
                start = 0
                db_bias = db_content_lbracket+1+col_lidx
                
                for db_idx in db_sep_index_list:
                    db_t5toks_list.append([i for i in range(start+db_bias, db_idx+db_bias)])
                    start = db_idx + 1
                col = col[:db_content_lbracket]
                col_ridx = col_lidx + len(col)
            col = "".join(col)
            
            if len(col) == 0:
                continue
            try:
                lge_column_idx = lge_column_ori.index(col)
                column_lgeid2t5id[lge_column_idx] = [i for i in range(col_lidx,col_ridx)]
                dbcontent_lgeid2dbt5id[lge_column_idx] = db_t5toks_list
                lge_column[lge_column_idx] = "[None]"
                lge_column_ori[lge_column_idx] = "[None]"

            except:
                t5_columns = col.split(",")
                for t5_col in t5_columns:
                    try:
                        lge_column_idx = lge_column_ori.index(t5_col)
                        column_lgeid2t5id[lge_column_idx] = [i for i in range(col_lidx,col_ridx)]
                        dbcontent_lgeid2dbt5id[lge_column_idx] = db_t5toks_list
                        lge_column[lge_column_idx] = "[None]"
                        lge_column_ori[lge_column_idx] = "[None]"
                    except:
                        if  col_ridx != len(t5_toks):
                            flag = False
                            continue

    res = {}
    res["column_lgeid2t5id"] = column_lgeid2t5id
    res["table_lgeid2t5id"] = table_lgeid2t5id
    res["dbcontent_lgeid2dbt5id"] = dbcontent_lgeid2dbt5id
    res["t5_toks_ids"] = t5_item
    res["t5_toks"] = t5_toks 
    res["graph"] = lge_item["graph"] 
    res["db_name"] = lge_item["db_name"] 
    res["ori_toks"] = lge_item["ori_toks"]
    return res, flag

def choose_dict_from_list(lge_q_num, lge_t_num, lge_c_num, id): 
    if id < lge_q_num:
        return 0, id, "question"
    elif id < lge_q_num + lge_t_num:
        return 1, id-lge_q_num, "table"
    else:
        return 2, id - (lge_q_num + lge_t_num), "column"


def generate_relations(res_item, table_lgesql):
    err_edge = 0
    all_edge = 0

    t5_toks_ids = res_item["t5_toks_ids"]
    lge_edges = res_item["graph"].global_edges
    relation = np.zeros((len(t5_toks_ids), len(t5_toks_ids)), dtype=int)
    db_name = res_item["db_name"]

    lge_q_num = len(res_item["ori_toks"])
    lge_t_num = len(table_lgesql[db_name]['table_names'])
    lge_c_num = len(table_lgesql[db_name]['column_names'])
    lge_toks = res_item["ori_toks"] + table_lgesql[db_name]['table_names'] + table_lgesql[db_name]['column_names']
    
    question_lgeid2t5id = res_item["question_lgeid2t5id"]
    table_lgeid2t5id = res_item["table_lgeid2t5id"]
    column_lgeid2t5id = res_item["column_lgeid2t5id"]
    dbcontent_lgeid2dbt5id = res_item["dbcontent_lgeid2dbt5id"]
    lgeid2t5id_list = [question_lgeid2t5id, table_lgeid2t5id, column_lgeid2t5id]
    
    for edge in lge_edges:
        all_edge += 1
        src_id, dst_id, r = edge
        r_id = relation2id_dict[r]
        src_dict_idx, new_src_id, src_kind = choose_dict_from_list(lge_q_num, lge_t_num, lge_c_num, src_id)
        src_dict = lgeid2t5id_list[src_dict_idx]
        
        if new_src_id not in src_dict.keys():
            if new_src_id < len(src_dict.keys()):
                err_edge += 1
            continue
        t5_src_id = src_dict[new_src_id]
        
        db_t5_src_id = []
        if src_dict_idx == 2 and new_src_id in dbcontent_lgeid2dbt5id.keys():
            db_t5_src_id = [i for item in dbcontent_lgeid2dbt5id[new_src_id] for i in item]
        
        dst_dict_idx, new_dst_id, dst_kind = choose_dict_from_list(lge_q_num, lge_t_num, lge_c_num, dst_id)
        dst_dict = lgeid2t5id_list[dst_dict_idx]
        if new_dst_id not in dst_dict.keys():
            if new_dst_id < len(dst_dict.keys()):
                err_edge += 1
            continue
        t5_dst_id = dst_dict[new_dst_id]
        
        db_t5_dst_id = []
        if dst_dict_idx == 2 and new_dst_id in dbcontent_lgeid2dbt5id.keys():
            db_t5_dst_id = [i for item in dbcontent_lgeid2dbt5id[new_dst_id] for i in item]

        db_t5_src_id = []
        db_t5_dst_id = []
        t5_src_id_list = [t5_src_id] if len(db_t5_src_id) == 0 else [t5_src_id, db_t5_src_id]
        t5_dst_id_list = [t5_dst_id] if len(db_t5_dst_id) == 0 else [t5_dst_id, db_t5_dst_id]
        
        for ii in range(len(t5_src_id_list)):
            for jj in range(len(t5_dst_id_list)):
                for pair_i, pair_j in itertools.product(t5_src_id_list[ii], t5_dst_id_list[jj]):
                    relation[pair_i][pair_j] = r_id
                    
    for node_idx in range(1, lge_c_num):
        if node_idx in column_lgeid2t5id.keys() and node_idx in dbcontent_lgeid2dbt5id.keys():
            col_t5id_list = column_lgeid2t5id[node_idx]
            dbcontent_t5id_list = dbcontent_lgeid2dbt5id[node_idx]
            for dbcontent_t5id in dbcontent_t5id_list:
                for pair_i, pair_j in itertools.product(col_t5id_list, dbcontent_t5id):
                    relation[pair_i][pair_j] = relation2id_dict['has-dbcontent']
                    relation[pair_j][pair_i] = relation2id_dict['has-dbcontent-r']

    return relation

def main_preprocessing_process():
    mode_list = ["train", "dev"]
    t5_processed = pickle.load(open("/home/jxqi/text2sql/data/0223_dataset_db_content_split.pkl", "rb"))# need to change
    train_lgesql = pickle.load(open("/home/jxqi/text2sql/data/lge_files/train_jytang.rgatsql.bin", "rb"))# need to change
    dev_lgesql = pickle.load(open("/home/jxqi/text2sql/data/lge_files/dev_jytang.rgatsql.bin", "rb"))# need to change
    table_lgesql = pickle.load(open("/home/jxqi/text2sql/data/lge_files/tables_jytang.bin", "rb"))# need to change
    for mode in mode_list:
        dataset_lgesql = train_lgesql if mode == "train" else dev_lgesql
        total_relations = []
        for i, lgeitem in tqdm(enumerate(dataset_lgesql[:min(7000, len(dataset_lgesql))])):
            t5_item = t5_processed[mode][i]["input_ids"] # need to change
            ques_res, _ = match_question(t5_item, lgeitem, mode)
            tab_col_res, _ = match_table_and_column(t5_item, lgeitem, table_lgesql)
            res_item = ques_res
            res_item.update(tab_col_res)
            relation = generate_relations(res_item, table_lgesql)
            total_relations.append(relation)
        pickle.dump(total_relations, open("/home/jxqi/text2sql/data/0223_" + mode +"_lge_update_relation.pickle", "wb"))
        print("save ", mode, " successfully!")


def preprocessing_dataset_split(dataset_split_input_ids, dataset_split_lgesql_filepath, table_lgesql_filepath, mode):
    mode_list = ["train", "dev"]
    # train_lgesql = pickle.load(open("/home/jxqi/text2sql/data/lge_files/train_jytang.rgatsql.bin", "rb"))# need to change
    dataset_split_lgesql = pickle.load(open(dataset_split_lgesql_filepath, "rb"))# need to change
    table_lgesql = pickle.load(open(table_lgesql_filepath, "rb"))# need to change
    # dev_lgesql = pickle.load(open("/home/jxqi/text2sql/data/lge_files/dev_jytang.rgatsql.bin", "rb"))# need to change
    # table_lgesql = pickle.load(open("/home/jxqi/text2sql/data/lge_files/tables_jytang.bin", "rb"))# need to change
    dataset_lgesql = dataset_split_lgesql
    total_relations = []
    for i, lgeitem in tqdm(enumerate(dataset_lgesql[:min(7000, len(dataset_lgesql))])):
        t5_item = dataset_split_input_ids[i] # need to change
        ques_res, _ = match_question(t5_item, lgeitem, mode)
        tab_col_res, _ = match_table_and_column(t5_item, lgeitem, table_lgesql)
        res_item = ques_res
        res_item.update(tab_col_res)
        relation = generate_relations(res_item, table_lgesql)
        total_relations.append(relation)

    return total_relations


if __name__ == "__main__":
    main_preprocessing_process()