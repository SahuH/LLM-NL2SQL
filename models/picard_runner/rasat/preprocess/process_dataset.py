#coding=utf8
import os, json, pickle, time
from tqdm import tqdm
import fcntl

from .common_utils import Preprocessor
from .align_tables import align_tables_by_dataset_name


def process_tables(processor, tables_list, output_path=None):
    tables = {}
    for idx, each in tqdm(enumerate(tables_list)):
        tables[each['db_id']] = processor.preprocess_database(each)
    print('In total, process %d databases .' % (len(tables)))
    if output_path is not None:
        with open (output_path, 'wb') as dump_f:
            fcntl.flock(dump_f.fileno(), fcntl.LOCK_EX)
            pickle.dump(tables, dump_f)
    return tables

def process_dataset(processor, dataset, tables, dataset_name, mode, output_path_base=None, used_coref=False):
    processed_dataset = []
    # if dataset_name == "cosql":
    #     wfile = open(f"./dataset_files/ori_dataset/cosql_dataset/sql_state_tracking/{mode}_text_list.txt", "w")
    if used_coref and not os.path.exists(os.path.join(output_path_base, f"{mode}_coref.json")):
        wfile = open(os.path.join(output_path_base, f"{mode}_text_list.txt"), "w")
    for idx, entry in tqdm(enumerate(dataset)):
        # if idx > 100:
        #     continue
        if "spider" in dataset_name:
            entry = processor.pipeline(entry, tables[entry['db_id']], dataset_name, idx)
        elif dataset_name in ["cosql", "sparc"]:
            entry = processor.pipeline(entry, tables[entry['database_id']], dataset_name, idx)
        else:
            raise NotImplementedError
        if used_coref and not os.path.exists(os.path.join(output_path_base, f"{mode}_coref.json")):
            wfile.write(str(entry['final_preprocessed_text_list'])+"\n")
        processed_dataset.append(entry)
    with open(os.path.join(output_path_base, f"{mode}.bin"), 'wb') as dump_f:
        fcntl.flock(dump_f.fileno(), fcntl.LOCK_EX)
        pickle.dump(processed_dataset, dump_f)
    return processed_dataset

def init_dataset_path(data_base_dir, dataset_name, mode):
    db_dir = os.path.join(data_base_dir, "ori_dataset", dataset_name, "database")
    table_data_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "tables.json")
    table_out_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.bin")
    if mode == "train":
        if dataset_name == "spider":
            dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "train_spider.json")
        elif dataset_name == "cosql":
            db_dir = os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "database")
            dataset_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset/sql_state_tracking/", "cosql_train.json")
            table_data_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "tables.json")
            
        elif dataset_name == "sparc":
            dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "train.json")
        else:
            raise NotImplementedError
        # dataset_output_path_base=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "train.bin")
    elif mode == "dev": 
        if dataset_name in ["spider", "sparc", "spider_desc"] :
            dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "dev.json")
        elif dataset_name == "cosql":
            db_dir = os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "database")
            dataset_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset/sql_state_tracking/", "cosql_dev.json")
            table_data_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "tables.json")
            
        elif dataset_name == "spider_dk":
            dataset_path = os.path.join(data_base_dir, "ori_dataset", dataset_name, "spider-DK.json")
        elif dataset_name == "spider_dates":
            dataset_path = os.path.join(data_base_dir, "ori_dataset", dataset_name, "dev_spider_dates.json")
        elif dataset_name == "dk_spider_dates":
            dataset_path = os.path.join(data_base_dir, "ori_dataset", dataset_name, "dk_spider_dates.json")
        elif dataset_name == "spider_syn":
            dataset_path = os.path.join(data_base_dir, "ori_dataset", dataset_name, "spider-syn.json")
        else:
            raise NotImplementedError
        # dataset_output_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "dev.bin")
    else:
        raise NotImplementedError
    dataset_output_path_base=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)
    if not os.path.exists(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)):
        os.makedirs(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name))
    return db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path_base

def preprocessing_generate_lgerels(data_base_dir, dataset_name, mode, used_coref = False, use_dependency=False):
    db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path_base = init_dataset_path(data_base_dir, dataset_name, mode)
    processor = Preprocessor(dataset_name, db_dir=db_dir, db_content=True)
    
    # loading database and dataset
    print(f"Dataset name: {dataset_name}")
    print(f"Mode: {mode}")
    if not os.path.exists(table_out_path):
        with open(table_data_path, 'r') as load_f: 
            fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
            tables_list = json.load(load_f)
        print('Firstly, preprocess the original databases ...')
        tables_list = align_tables_by_dataset_name(dataset_name, tables_list)
        print('Tables alignments done...')
        start_time = time.time()
        tables = process_tables(processor, tables_list, table_out_path)
        print('Databases preprocessing costs %.4fs .' % (time.time() - start_time))
    else:
        tables = pickle.load(open(table_out_path, 'rb'))
        print('Databases has been preprocessed. Use cache.')

    with open(dataset_path, 'r') as load_f: 
        fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
        dataset = json.load(load_f)
    start_time = time.time()

    if not os.path.exists(os.path.join(dataset_output_path_base, f"{mode}.bin")):
        dataset = process_dataset(processor, dataset, tables, dataset_name, mode, dataset_output_path_base, used_coref)
        print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
    else:
        print('Dataset has been preprocessed. Use cache.')
