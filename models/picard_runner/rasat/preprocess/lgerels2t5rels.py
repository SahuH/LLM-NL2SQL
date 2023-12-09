import enum
from json.tool import main
import pickle
from tkinter.tix import MAIN
from tokenize import group
import numpy as np
import itertools
import os
import fcntl
import json

from tqdm import tqdm
from transformers import AutoTokenizer
from tokenizers import AddedToken
from .transform_utils import mul_mul_match, get_idx_list, find_sep_mullen, find_all_sep_index_from_list, find_all_sep_pair_from_list, raise_key, merge_two_dict, decode_from_dict, decode_from_pair_dict, tokid2sent
from .constants import MAX_RELATIVE_DIST
from .get_relation2id_dict import get_relation2id_dict



def match_question(t5_processed, dataset_lgesql, t5_tokenizer, lge_tokenizer, dataset_name,  mode):
    err = 0
    total_example = 0
    t5_dataset_idx = 0
    for lge_dataset_idx in tqdm(range(len(dataset_lgesql))):
        lge_aux_question_idx_list = get_idx_list(dataset_lgesql[lge_dataset_idx], dataset_name)
        for j, lge_aux_question_idx in enumerate(lge_aux_question_idx_list):
            t5_toks_ids = t5_processed[t5_dataset_idx]
            t5_dataset_idx += 1
            t5_toks = []
            for id in t5_toks_ids:
                w = t5_tokenizer.decode(id).replace("</s>", "")
                t5_toks.append(w.replace(" ", ""))
            aux_sep = find_sep_mullen(t5_toks, "|")
            aux_text = t5_toks[aux_sep[0]+2:]
            mul_idx = aux_sep[0]+2
            aux_sep_list = find_all_sep_index_from_list(aux_text, "|")
            aux_text_list = []
            aux_start = 0
            for aux_sep in aux_sep_list:
                aux_text_list.append((aux_text[aux_start:aux_sep], mul_idx+aux_start))
                aux_start = aux_sep + 1
            for k, question_idx in enumerate(lge_aux_question_idx):
                total_example += 1
                question_lgeid2t5id = {}
                t5_bias = 0
                lge_r_question_toks = dataset_lgesql[lge_dataset_idx][f'ori_toks_{question_idx}']
                if k == 0:
                    sep = t5_toks.index("|")
                    t5_toks_k = t5_toks[:sep]
                    t5_bias = 0
                else:
                    try:
                        t5_toks_k = aux_text_list[k-1][0]
                        t5_bias = aux_text_list[k-1][1]
                    except:
                        if (len(t5_toks)<512):
                            err += 1
                        continue
                t5_toks_k = ([subword.replace("▁", "") for subword in t5_toks_k])
                lge_r_question_toks = [lge_tokenizer.decode(lge_tokenizer.encode(r_question_toks)[:-1]) for r_question_toks in lge_r_question_toks]
                start = 0
                toks_idx = 0
                while toks_idx < len(lge_r_question_toks) and start < len(t5_toks_k):
                    append_t5_idx, append_q_idx = mul_mul_match(t5_toks[t5_bias + start:],  lge_r_question_toks[toks_idx:])
                    match_t5_id_list = [i for i in range(t5_bias+start,t5_bias+start+append_t5_idx)]
                    match_q_id_list = [i for i in range(toks_idx,toks_idx+append_q_idx)]
                    for q_idx in match_q_id_list:
                        question_lgeid2t5id[q_idx] = match_t5_id_list
                    if append_t5_idx == -1 and append_q_idx == -1:
                        if len(t5_toks) < 512:
                            err+=1
                            print(lge_aux_question_idx_list)
                            print(t5_dataset_idx, t5_toks_k[start:])
                            print(lge_dataset_idx, lge_r_question_toks[toks_idx:])
                            print(t5_tokenizer.decode(t5_toks_ids))
                            print(lge_r_question_toks)
                            print(t5_toks_k)     
                            print(dataset_lgesql[lge_dataset_idx]['processed_text_list'])
                            print()
                        break
                    else:
                        start += append_t5_idx
                        toks_idx += append_q_idx
                # wfile.write(str(lge_dataset_idx)+"\n"+str(lge_aux_question_idx_list)+"\n")
                # wfile.write(" ".join(lge_r_question_toks)+"\n")
                # wfile.write(" ".join(t5_toks)+"\n\n")
                dataset_lgesql[lge_dataset_idx][f"question_lgeid2t5id_{j}#{question_idx}"] = question_lgeid2t5id
                dataset_lgesql[lge_dataset_idx]["idx_list"] = lge_aux_question_idx_list
            dataset_lgesql[lge_dataset_idx][f"t5_toks_{j}"] = (t5_dataset_idx-1,t5_toks,t5_toks_ids)
    print(f"Question match errors: {err}/{total_example}")



def match_table_and_column(dataset_lgesql, table_lgesql, t5_tokenizer):
    err = 0
    total_example = 0
    for lge_dataset_idx,item in tqdm(enumerate(dataset_lgesql)):
        lge_aux_question_idx_list = dataset_lgesql[lge_dataset_idx]["idx_list"]
        for j, lge_aux_question_idx in enumerate(lge_aux_question_idx_list):
            column_lgeid2t5id = {}
            table_lgeid2t5id = {}
            dbcontent_lgeid2dbt5id = {}
            t5_toks_ids = dataset_lgesql[lge_dataset_idx][f"t5_toks_{j}"][2]
            t5_toks = []
            for id in t5_toks_ids:
                w = t5_tokenizer.decode(id).replace("</s>", "")
                t5_toks.append(w)
                t5_toks = ([subword for subword in t5_toks])

            mul_idx = (find_sep_mullen(t5_toks, '|'))[0]
            if mul_idx != len(t5_toks):
                t5_toks = t5_toks[:mul_idx]

            sep_index_list = find_all_sep_index_from_list(t5_toks, "|")

            db_name = "".join([w for w in t5_toks[sep_index_list[0]+1:sep_index_list[1]]])
            dataset_lgesql[lge_dataset_idx][f"db_name_{j}"] = db_name
            column_lgeid2t5id[0] = [i for i in range(sep_index_list[0]+1, sep_index_list[1])]

            lge_table = table_lgesql[db_name]['table_names']
            lge_table = [i.replace(" ", "_") for i in lge_table]
            lge_column = table_lgesql[db_name]['column_names']
            lge_column = [i[1].replace(" ", "_") for i in lge_column]
            lge_column_index = [i[0] for i in lge_column]

            lge_table_ori = [item.lower() for item in table_lgesql[db_name]['table_names_original']]
            lge_column_ori = [item[1].replace(" ", "").lower() for item in table_lgesql[db_name]['column_names_original']]
            lge_column_ori_index = [i[0] for i in table_lgesql[db_name]['column_names_original']]

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
                    lge_table[lge_tb_idx] = "[None]"
                    # lge_table_ori[lge_tb_idx] = "[None]"
                except:
                    try:
                        lge_tb_idx = lge_table_ori.index(table)
                        # lge_table[lge_tb_idx] = "[None]"
                        lge_table_ori[lge_tb_idx] = "[None]"
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
            if (not flag):
                # print(repr(col))
                # print(column_sep_index_list)
                # print(t5_toks[col_lidx: col_ridx])
                # print(" ".join(t5_toks))
                # print(lge_column)
                # print(lge_column_ori)
                # print(column_lgeid2t5id)
                err += 1
            total_example += 1
            dataset_lgesql[lge_dataset_idx][f"column_lgeid2t5id_{j}"] = column_lgeid2t5id
            dataset_lgesql[lge_dataset_idx][f"table_lgeid2t5id_{j}"] = table_lgeid2t5id
            dataset_lgesql[lge_dataset_idx][f"dbcontent_lgeid2dbt5id_{j}"] = dbcontent_lgeid2dbt5id  
    print(f"DB match errors: {err}/{total_example}")

def generate_relations_between_questions(relation, lge_aux_question_idx, dataset_lgesql_item, RELATION2ID_DICT, j):
    question_t5_id_dict = {}
    for k, question_idx in enumerate(lge_aux_question_idx):
        if f"question_lgeid2t5id_{j}#{question_idx}" not in dataset_lgesql_item.keys() or len(dataset_lgesql_item[f"question_lgeid2t5id_{j}#{question_idx}"])==0:
            continue
        question_lgeid2t5id = dataset_lgesql_item[f"question_lgeid2t5id_{j}#{question_idx}"]
        t5_id_list = sorted([t5_ids  for t5_ids in question_lgeid2t5id.values()])
        question_t5_id_dict[question_idx] = t5_id_list
    for id_list_i in question_t5_id_dict.values():
        for id_list_j in question_t5_id_dict.values():
            min_i, max_i = id_list_i[0][0], id_list_i[-1][-1]
            min_j, max_j = id_list_j[0][0], id_list_j[-1][-1]
            relation[min_i:max_i+1, min_j:max_j+1]=RELATION2ID_DICT["question-question-generic"]
    for question_idx in range(1, len(lge_aux_question_idx)):
        if (question_idx-1 not in question_t5_id_dict.keys() or question_idx not in question_t5_id_dict):
            continue
        last_t5_ids = question_t5_id_dict[question_idx-1][-MAX_RELATIVE_DIST:]
        pre_t5_ids = question_t5_id_dict[question_idx][:MAX_RELATIVE_DIST]
        for last_t5_idx in range(len(last_t5_ids)):
            for pre_t5_idx in range(len(pre_t5_ids)):
                distance = pre_t5_idx + len(last_t5_ids)-last_t5_idx
                if distance <= MAX_RELATIVE_DIST:
                    for pair_i, pair_j in itertools.product(last_t5_ids[last_t5_idx], pre_t5_ids[pre_t5_idx]):
                        relation[pair_i][pair_j] = RELATION2ID_DICT[f"question-question-dist{distance}"]
                        relation[pair_j][pair_i] = RELATION2ID_DICT[f"question-question-dist{-distance}"]


def remove_notused_coref(lge_aux_question_idx, coref_dataset):
    used_coref_dataset = {}
    
    # print(coref_dataset)
    for group_key in coref_dataset["coref"].keys():
        new_group_list = []
        for group_item_list in coref_dataset["coref"][group_key]["group"]:
            new_group_item_list = []
            for item in group_item_list:
                if item["turn"] in lge_aux_question_idx:
                    new_group_item_list.append(item)
            if len(new_group_item_list) >= 1:
                new_group_list.append(new_group_item_list)

        if len(new_group_list) >= 2:
            used_coref_dataset[group_key] = new_group_list
        # used_set = coref_dataset["coref"][group_key]["used_turn"]
        # for turn in used_set:
        #     if turn not in lge_aux_question_idx:
        #         flag = False
        #         break
        # if flag:    
        #     used_coref_dataset[group_key] = coref_dataset["coref"][group_key]
    # print(coref_dataset)
    # print(lge_aux_question_idx)
    # print(used_coref_dataset)
    # print()
    return used_coref_dataset


def generate_coref_relations(relation, coref_dataset, cur_dataset_lgesql, j, RELATION2ID_DICT):
    for group in coref_dataset.keys():
        coref_relation_t5id_list = []
        for coref_li in coref_dataset[group]:
            co_relation_t5id_list = []
            for coref_item in coref_li:
                if (f"question_lgeid2t5id_{j}#{coref_item['turn']}" not in cur_dataset_lgesql.keys()) or len(cur_dataset_lgesql[f"question_lgeid2t5id_{j}#{coref_item['turn']}"].keys())==0:
                    continue
                
                question_lgeid2t5id = cur_dataset_lgesql[f"question_lgeid2t5id_{j}#{coref_item['turn']}"] 

                if coref_item["position"] not in question_lgeid2t5id.keys():
                    continue
                t5_id = question_lgeid2t5id[coref_item["position"]]
                co_relation_t5id_list.append(t5_id)
            coref_relation_t5id_list.append([_ for item in co_relation_t5id_list for _ in item])
            # print(co_relation_t5id_list)
            if len(co_relation_t5id_list) > 1:
                for pair_i, pair_j in itertools.combinations(co_relation_t5id_list, 2):
                    relation[pair_i][pair_j] = RELATION2ID_DICT["co_relations"]
                    relation[pair_j][pair_i] = RELATION2ID_DICT["co_relations"]
        
        for ii in range(len(coref_relation_t5id_list)): 
            for jj in range(ii+1, len(coref_relation_t5id_list)):
                for pair_i, pair_j in itertools.product(coref_relation_t5id_list[ii], coref_relation_t5id_list[jj]):
                    relation[pair_i][pair_j] = RELATION2ID_DICT["coref_relations"]

            



def generate_relations(dataset_lgesql, t5_processed, table_lgesql, RELATION2ID_DICT, edgeType, t5_tokenizer, dataset_name, coref_dataset, mode):
    err_edge = 0
    total_edge = 0
    t5_dataset_idx = 0
    res_relations = []

    for lge_dataset_idx in tqdm(range(len(dataset_lgesql))):
        lge_aux_question_idx_list = dataset_lgesql[lge_dataset_idx]["idx_list"]
        for j, lge_aux_question_idx in enumerate(lge_aux_question_idx_list):
            
            

            t5_toks_ids = dataset_lgesql[lge_dataset_idx][f"t5_toks_{j}"][2]
            t5_dataset_idx = dataset_lgesql[lge_dataset_idx][f"t5_toks_{j}"][0]
            db_name = dataset_lgesql[lge_dataset_idx][f"db_name_{j}"]
            
            relation = np.zeros((len(t5_toks_ids), len(t5_toks_ids)), dtype=int)
            
            generate_relations_between_questions(relation, lge_aux_question_idx, dataset_lgesql[lge_dataset_idx], RELATION2ID_DICT, j)
            
            table_lgeid2t5id = dataset_lgesql[lge_dataset_idx][f"table_lgeid2t5id_{j}"]
            column_lgeid2t5id = dataset_lgesql[lge_dataset_idx][f"column_lgeid2t5id_{j}"]
            dbcontent_lgeid2dbt5id = dataset_lgesql[lge_dataset_idx][f"dbcontent_lgeid2dbt5id_{j}"]
            
            lge_t_num = len(table_lgesql[db_name]['table_names'])
            lge_c_num = len(table_lgesql[db_name]['column_names'])
            
            dbcontent_lgeid2dbt5id_raise = raise_key(dbcontent_lgeid2dbt5id, lge_t_num)
            schema_lgeid2t5id = merge_two_dict(table_lgeid2t5id, column_lgeid2t5id, lge_t_num)

            if coref_dataset is not None:
                used_coref_dataset = remove_notused_coref(lge_aux_question_idx, coref_dataset[lge_dataset_idx])
                if len(used_coref_dataset.keys()) > 0:
                    generate_coref_relations(relation, used_coref_dataset, dataset_lgesql[lge_dataset_idx], j, RELATION2ID_DICT)

            for k, question_idx in enumerate(lge_aux_question_idx):
                if (f"question_lgeid2t5id_{j}#{question_idx}" not in dataset_lgesql[lge_dataset_idx].keys()):
                    if len(t5_toks_ids) < 512:
                        err_edge += 1
                        tokid2sent(t5_tokenizer, t5_toks_ids)
                    continue
                
                question_lgeid2t5id = dataset_lgesql[lge_dataset_idx][f"question_lgeid2t5id_{j}#{question_idx}"]
                
                if "Dependency" in edgeType:
                    qq_relations = dataset_lgesql[lge_dataset_idx][f"tree_relations_{question_idx}"]
                else:
                    qq_relations = dataset_lgesql[lge_dataset_idx][f"relations_{question_idx}"]
                ss_relations = table_lgesql[dataset_lgesql[lge_dataset_idx]["database_id"]]["relations"] if dataset_name in ["cosql", "sparc"] else table_lgesql[dataset_lgesql[lge_dataset_idx]["db_id"]]["relations"]
                qs_relations = dataset_lgesql[lge_dataset_idx][f"schema_linking_{question_idx}"][0]
                sq_relations = dataset_lgesql[lge_dataset_idx][f"schema_linking_{question_idx}"][1]

                relation_list = [qq_relations, ss_relations, qs_relations, sq_relations]
                relative_id_list = [(question_lgeid2t5id, question_lgeid2t5id, (0, 0)), (schema_lgeid2t5id, schema_lgeid2t5id, (1, 1)), (question_lgeid2t5id, schema_lgeid2t5id, (0, 1)), (schema_lgeid2t5id, question_lgeid2t5id, (1, 0))]
                for relation_list_idx, relations in enumerate(relation_list):
                    edges = [(i, j, relations[i][j]) for i  in range(len(relations)) for j in range(len(relations[0]))]
                    for edge in edges:
                        total_edge += 1
                        try:
                            if edge[2] in ["question-question-generic"]:
                                continue
                            t5_src_id = relative_id_list[relation_list_idx][0][edge[0]]
                            if (relative_id_list[relation_list_idx][2][0] == 1 and edge[0] in dbcontent_lgeid2dbt5id_raise.keys()):
                                db_t5_src_id = [i for item in dbcontent_lgeid2dbt5id_raise[edge[0]] for i in item]
                                
                            t5_dst_id = relative_id_list[relation_list_idx][1][edge[1]]
                            if (relative_id_list[relation_list_idx][2][1] == 1 and edge[1] in dbcontent_lgeid2dbt5id_raise.keys()):
                                db_t5_dst_id = [i for item in dbcontent_lgeid2dbt5id_raise[edge[1]] for i in item]
                                
                            r_id = RELATION2ID_DICT[edge[2]]
                        except Exception as e:
                            if len(t5_toks_ids) < 512:
                                # print(t5_dataset_idx)
                                # print(e)
                                # tokid2sent(t5_tokenizer, t5_toks_ids)
                                # print(relation_list_idx)
                                # print(edge[0])
                                # print(relative_id_list[relation_list_idx][0].keys())
                                # print(edge[1])
                                # print(relative_id_list[relation_list_idx][1].keys())
                                # print(table_lgeid2t5id.keys(), column_lgeid2t5id.keys())
                                # decode_from_dict(t5_tokenizer, table_lgeid2t5id, t5_toks_ids)
                                # decode_from_dict(t5_tokenizer, column_lgeid2t5id, t5_toks_ids)
                                # decode_from_pair_dict(t5_tokenizer, dbcontent_lgeid2dbt5id, t5_toks_ids)
                                # print()
                                err_edge+=1
                                break
                        db_t5_src_id = []
                        db_t5_dst_id = []
                        t5_src_id_list = [t5_src_id] if len(db_t5_src_id) == 0 else [t5_src_id, db_t5_src_id]
                        t5_dst_id_list = [t5_dst_id] if len(db_t5_dst_id) == 0 else [t5_dst_id, db_t5_dst_id]

                        for ii in range(len(t5_src_id_list)): #debug
                            for jj in range(len(t5_dst_id_list)):
                                for pair_i, pair_j in itertools.product(t5_src_id_list[ii], t5_dst_id_list[jj]):
                                    relation[pair_i][pair_j] = r_id
                for node_idx in range(1, lge_c_num):
                    if node_idx in column_lgeid2t5id.keys() and node_idx in dbcontent_lgeid2dbt5id.keys():
                        col_t5id_list = column_lgeid2t5id[node_idx]
                        dbcontent_t5id_list = dbcontent_lgeid2dbt5id[node_idx]
                        for dbcontent_t5id in dbcontent_t5id_list:
                            for pair_i, pair_j in itertools.product(col_t5id_list, dbcontent_t5id):
                                relation[pair_i][pair_j] = RELATION2ID_DICT['has-dbcontent']
                                relation[pair_j][pair_i] = RELATION2ID_DICT['has-dbcontent-r']
            res_relations.append(relation)
    print(f"Edge match errors: {err_edge}/{total_edge}")
    return t5_dataset_idx, res_relations


def init_tokenizer():
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
    lge_tokenizer = AutoTokenizer.from_pretrained('t5-small')
    t5_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    lge_tokenizer.add_tokens([AddedToken("<="), AddedToken("<")])
    return t5_tokenizer, lge_tokenizer

def init_dataset(data_base_dir, dataset_name, mode):
    table_lgesql=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.bin")
    if mode == "train":
        dataset_lgesql=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "train.bin")
    elif mode == "dev":   
        dataset_lgesql=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "dev.bin")
    else:
        raise NotImplementedError
    with open(table_lgesql, "rb") as load_f:
        fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
        table_lgesql = pickle.load(load_f)
    with open(dataset_lgesql, "rb") as load_f:
        fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
        dataset_lgesql = pickle.load(load_f)
    return dataset_lgesql, table_lgesql

def preprocessing_lgerels2t5rels(data_base_dir, dataset_name, t5_processed, mode, edgeType="Default", use_coref = False):
    t5_tokenizer, lge_tokenizer = init_tokenizer()
    dataset_lgesql, table_lgesql = init_dataset(data_base_dir, dataset_name, mode)
    RELATION2ID_DICT = get_relation2id_dict(edgeType, use_coref)

    print(f"Dataset: {dataset_name}")
    print(f"Mode: {mode}")
    print("Match Questions...")
    match_question(t5_processed, dataset_lgesql, t5_tokenizer, lge_tokenizer, dataset_name, mode)
    print("Match Table, Columns, DB Contents...")
    match_table_and_column(dataset_lgesql, table_lgesql, t5_tokenizer)
    print("Generate Relations...")
    
    if use_coref:
        with open(f"./dataset_files/preprocessed_dataset/{dataset_name}/{mode}_coref.json", 'r') as load_f: 
            fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
            coref_dataset = json.load(load_f)
    else:
        coref_dataset = None
    last_t5_dataset_idx, relations = generate_relations(dataset_lgesql, t5_processed, table_lgesql, RELATION2ID_DICT, edgeType, t5_tokenizer, dataset_name, coref_dataset, mode)
    # with open(f"{mode}.pickle", "wb") as load_f:
    #     fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
    #     pickle.dump(relations, load_f)   
    return last_t5_dataset_idx, relations
