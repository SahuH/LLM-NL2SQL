from tqdm import tqdm

def rough_look(load_dict):
    for idx, table_lgesql in tqdm(enumerate(load_dict)):
        ori_table = table_lgesql['table_names_original']
        table = table_lgesql['table_names']
        ori_col = [item for item in table_lgesql['column_names_original']]
        col = [item for item in table_lgesql['column_names']]
        diff_table = set([(ori_table[idx], table[idx]) for idx in range(len(ori_table)) if ori_table[idx].lower().replace("_", " ")!=table[idx] and ori_table[idx].lower().replace("_", " ") in table])
        diff_col = set([(ori_col[idx], col[idx]) for idx in range(len(ori_col)) if ori_col[idx][1].lower().replace("_", " ")!=col[idx][1] and (ori_col[idx][1].lower().replace("_", " ") in col)])
        if len(diff_table)>0:
            print(idx)
            print(diff_table)
            print(table)
            print(ori_table)
            print()
        if len(diff_col)>0:
            print(diff_col)
            print(col)
            print(ori_col)
            print()

def isSame( ori_text, text):
    if ori_text.lower().replace("_", " ")==text:
        return True
    if text.replace(" ", "").lower() == ori_text.lower():
        return True
    return False

def fine_look(load_dict):
    for idx, table_lgesql in tqdm(enumerate(load_dict)):
    #     print(table_lgesql)
        ori_table = table_lgesql['table_names_original']
        table = table_lgesql['table_names']
        ori_col = [(item[0], item[1], idx) for idx, item in enumerate(table_lgesql['column_names_original'])]
        col = [item for item in table_lgesql['column_names']]
        diff_table = set([(ori_table[idx], table[idx]) for idx in range(len(ori_table)) if not isSame(ori_table[idx],table[idx])])
        diff_col_name = set([(str(ori_col[idx][1]), str(col[idx][1])) for idx in range(len(ori_col)) if not isSame(ori_col[idx][1], col[idx][1])])

        # if len(diff_table)>0:
        #     print(idx)
        #     print(diff_table)
        #     print(ori_table)
        #     print(table)
        #     print()
        if len(diff_col_name)>0:
            print(idx)
            print(diff_col_name)
            print(ori_col)
            print(col)
            print()
        # if len(diff_col_id)>0:
        #     print(idx)
        #     print(diff_col_id)
        #     print(col)
        # if len(diff_col)>0:
        #     print(idx)
        #     print(diff_col)
        #     print(ori_col)
        #     print(col)
        #     print()
    #         print(ori_col)
            # print()


def get_column_compact_list(load_dict_item):
    # index start from -1
    max_idx = load_dict_item[-1][0]
    compact_list = [[] for i in range(max_idx+2)]
    for item in load_dict_item:
        compact_list[item[0]+1].append(item)  
    return compact_list

def recovery_from_compact_list(compact_list):
    res_list = []
    for i, list in enumerate(compact_list):
        for j, item in enumerate(compact_list[i]):
            res_list.append([i-1, item[1]])
    return res_list       


def modify_tables_spider(load_dict):
    for idx, item in enumerate(load_dict):
        if item["db_id"] == "store_1":
            diff_item = item["table_names"]
            diff_item[0], diff_item[1] = diff_item[1], diff_item[0]

            diff_item = item["column_names"]
            diff_compact_list = get_column_compact_list(diff_item)
            pop_item = diff_compact_list.pop(1)
            diff_compact_list.insert(2, pop_item)
            item["column_names"] = recovery_from_compact_list(diff_compact_list)
            # for pair in zip(item["column_names"], item["column_names_original"]):
            #     print(pair)
    return load_dict

def modify_tables_cosql(load_dict):
    for idx, item in enumerate(load_dict):
        if item["db_id"] == "scholar":
            diff_item = item["table_names"]
            pop_item = diff_item.pop(1)
            diff_item.insert(5, pop_item)
            pop_item = diff_item.pop(-2)
            diff_item.insert(0, pop_item)

            diff_item = item["column_names"]
            diff_compact_list = get_column_compact_list(diff_item)
            pop_item = diff_compact_list.pop(-2)
            diff_compact_list.insert(1, pop_item)
            pop_item = diff_compact_list.pop(3)
            diff_compact_list.insert(7, pop_item)
            item["column_names"] = recovery_from_compact_list(diff_compact_list)

        elif item["db_id"] == "store_1":
            diff_item = item["table_names"]
            diff_item[0], diff_item[1] = diff_item[1], diff_item[0]

            diff_item = item["column_names"]
            diff_compact_list = get_column_compact_list(diff_item)
            pop_item = diff_compact_list.pop(1)
            diff_compact_list.insert(2, pop_item)
            item["column_names"] = recovery_from_compact_list(diff_compact_list)

        elif item["db_id"] == "formula_1":
            diff_item = item["table_names"]
            pop_item = diff_item.pop(-3)
            diff_item.insert(0, pop_item)

            diff_item = item["column_names"]
            diff_compact_list = get_column_compact_list(diff_item)
            pop_item = diff_compact_list.pop(11)
            diff_compact_list.insert(1, pop_item)
            item["column_names"] = recovery_from_compact_list(diff_compact_list)
            

        elif item["db_id"] == "league_2":
            diff_item = item["table_names"]
            pop_item = diff_item.pop(-1)
            diff_item.insert(2, pop_item)

            diff_item = item["column_names"]
            diff_compact_list = get_column_compact_list(diff_item)
            pop_item = diff_compact_list.pop(-1)
            diff_compact_list.insert(2, pop_item)
            item["column_names"] = recovery_from_compact_list(diff_compact_list)

    return load_dict
    
def modify_tables_sparc(load_dict):  
    for idx, item in enumerate(load_dict):
        if item["db_id"] == "scholar":
            diff_item = item["table_names"]
            pop_item = diff_item.pop(1)
            diff_item.insert(5, pop_item)
            pop_item = diff_item.pop(-2)
            diff_item.insert(0, pop_item)

            diff_item = item["column_names"]
            diff_compact_list = get_column_compact_list(diff_item)
            pop_item = diff_compact_list.pop(-2)
            diff_compact_list.insert(1, pop_item)
            pop_item = diff_compact_list.pop(3)
            diff_compact_list.insert(7, pop_item)
            item["column_names"] = recovery_from_compact_list(diff_compact_list)

        elif item["db_id"] == "store_1":
            diff_item = item["table_names"]
            diff_item[0], diff_item[1] = diff_item[1], diff_item[0]

            diff_item = item["column_names"]
            diff_compact_list = get_column_compact_list(diff_item)
            pop_item = diff_compact_list.pop(1)
            diff_compact_list.insert(2, pop_item)
            item["column_names"] = recovery_from_compact_list(diff_compact_list)

        elif item["db_id"] == "formula_1":
            diff_item = item["table_names"]
            pop_item = diff_item.pop(-3)
            diff_item.insert(0, pop_item)

            diff_item = item["column_names"]
            diff_compact_list = get_column_compact_list(diff_item)
            pop_item = diff_compact_list.pop(11)
            diff_compact_list.insert(1, pop_item)
            item["column_names"] = recovery_from_compact_list(diff_compact_list)

    return load_dict
 
def align_tables_by_dataset_name(dataset_name, tables):
    if "spider" in dataset_name:
        tables = modify_tables_spider(tables)
    elif dataset_name == "cosql":
        tables = modify_tables_cosql(tables)
    elif dataset_name == "sparc":
        tables = modify_tables_sparc(tables)
    else:
        raise NotImplementedError
    return tables
