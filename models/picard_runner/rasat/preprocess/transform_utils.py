import pickle
import numpy as np
import functools
import itertools

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

def mul_mul_match_changeOrder(t5_toks_list,  question_toks_list):
    """"match two list of question toks"""
    t5_index = [i for i in range(0, len(t5_toks_list))]
    t5_index.reverse()
    question_index = [i for i in range(0, len(question_toks_list))]
    question_index.reverse()
    index_pair = list(itertools.product(t5_index, question_index))
    for i, j in index_pair:
        t5_toks = "".join(t5_toks_list[i:])
        question_toks = "".join(question_toks_list[j:])
        if t5_toks == question_toks:
            return i, j
    return -1,-1

def cmp(str1, str2):
    l1 = str1.split('#')
    l2 = str2.split('#')
    if (int(l1[0]!=l2[0])):
        return -1 if int(l1[0])<int(l2[0]) else 1 
    else:
        return -1 if int(l1[1])<int(l2[1]) else 1
        

def get_idx_list(res_dict, dataset_name):
    if dataset_name in ["cosql", "sparc"]:
        key = res_dict.keys()
        key = [k.split("_")[-1] for k in key if "relations" in k and "tree" not in k]
        key.sort(key = functools.cmp_to_key(cmp))
        total_res = [[key[0]]]
        tmp = []
        for i in range(1, len(key)):
            tmp.insert(0, key[i])
            if ("#" in key[i] and i+1 < len(key) and key[i].split('#')[0] == key[i+1].split('#')[0]):
                continue
            a = tmp.copy()
            total_res.append(a)
    else:
        total_res = [['-1']]
    return total_res

def get_idx_list_changeOrder(res_dict, dataset_name):
    if dataset_name in ["cosql", "sparc"]:
        key = res_dict.keys()
        key = [k.split("_")[-1] for k in key if "relations" in k and "tree" not in k]
        key.sort(key = functools.cmp_to_key(cmp))
        total_res = [[key[0]]]
        tmp = []
        for i in range(1, len(key)):
            tmp.append(key[i])
            if ("#" in key[i] and i+1 < len(key) and key[i].split('#')[0] == key[i+1].split('#')[0]):
                continue
            a = tmp.copy()
            total_res.append(a)
    else:
        total_res = [['-1']]
    return total_res

def isValid(idx, maxlen):
    if idx >=0 and idx<maxlen:
        return True
    return False

def find_sep_mullen(item_list, sep_item):
    start = 0
    times = 2
    sep_list = []
    while start < len(item_list):
        try:
            index = item_list.index(sep_item, start)
            start = index+1
            if isValid(start, len(item_list)):
                if (item_list[start] == sep_item):
                    sep_list.append(index)
        except:
            break
    sep_list.append(len(item_list))
    return sep_list

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

def raise_key(ori_dict, add_num):
    res_dict = {}
    for ori_key in ori_dict.keys():
        new_key = ori_key + add_num
        res_dict[new_key] = ori_dict[ori_key]
    return res_dict

def merge_two_dict(ori_dict, change_dict, add_num):
    res_dict = {}
    res_dict.update(ori_dict)
    res_dict.update(raise_key(change_dict, add_num))
    return res_dict

def decode_from_dict(t5_tokenizer, d, t5_toks_ids):
    v = [t5_toks_ids[i]  for item in d.values() for i in item]
    print(t5_tokenizer.decode(v).replace("</s>", ""))

def decode_from_pair_dict(t5_tokenizer, d, t5_toks_ids):
    v = [t5_toks_ids[id]  for items in d.values() for pair in items for id in pair]
    if len(v) > 0:
        print(t5_tokenizer.decode(v).replace("</s>", ""))
        print(t5_tokenizer.decode(t5_toks_ids).replace("</s>", ""))

def tokid2sent(t5_tokenizer, t5_toks_id):
    print(t5_tokenizer.decode(t5_toks_id).replace("</s>", ""))
