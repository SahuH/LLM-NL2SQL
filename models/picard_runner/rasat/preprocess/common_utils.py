#coding=utf8
import os, sqlite3
import numpy as np
import stanza
from nltk.corpus import stopwords
from itertools import product, combinations
import re

from .constants import MAX_RELATIVE_DIST

class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.children_relation = dict()
        self._size = -1
        self._depth = -1
        self.parents = []


    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)
    
    def add_parent(self, parent):
        self.parents.extend(parent)

    def add_relation(self, child_node, relation):
        self.children_relation[child_node] = relation

    def size(self):
        if getattr(self, '_size') != -1:
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth') != -1:
            return self._depth
        count = 1
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# def quote_normalization(question):
#     """ Normalize all usage of quotation marks into a separate \" """
#     new_question, quotation_marks = [], ['``', "''"]
#     re_patten = re.compile(r"'.+")
#     for idx, tok in enumerate(question):
#         for mark in quotation_marks:
#             tok = tok.replace(mark, "\"")
#         if re.match(re_patten, tok):
#             if len(tok) == 2 and not ( idx+1<len(question) and question[idx+1]=="'"):
#                 pass
#             else:
#                 tok = tok.replace("'", "' ")
#                 print(question)
#                 print(tok)
#         new_question.append(tok)
#     return new_question

# def quote_normalization(dataset_name, data_idx, question):
#     """ Normalize all usage of quotation marks into a separate \" """
#     new_question, quotation_marks, change_marks = [], ["'", '"', '`', '‘', '’', '“', '”', "‘‘", "’’"], ['``', "''"]
#     for idx, tok in enumerate(question):
#         for mark in change_marks:
#             tok = tok.replace(mark, "\"")
#         if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
#             new_question += [tok[0], tok[1:-1], tok[-1]]
#         elif len(tok) > 2 and tok[0] in quotation_marks:
#             new_question += [tok[0], tok[1:]]
#         elif len(tok) > 2 and tok[-1] in quotation_marks:
#             new_question += [tok[:-1], tok[-1]]
#         elif tok in quotation_marks:
#             new_question.append(tok)
#         elif len(tok) == 2 and tok[0] in quotation_marks:
#             # special case: the length of entity value is 1
#             if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
#                 new_question += [tok[0], tok[1]]
#             else:
#                 new_question.append(tok)
#         else:
#             new_question.append(tok)
    
#     print(new_question)
#     return new_question

def quote_normalization(dataset_name, data_idx, question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks, change_marks = [], ["'", '"', '`', '‘', '’', '“', '”', "‘‘", "’’"], ['``', "''"]
    idx = 0
    while idx < len(question):
        tok=question[idx]
        # if dataset_name == "cosql" and data_idx == 441 and tok=="''voluptatem":
        #     new_question.append(tok) 
        #     idx += 1
        #     new_question.append(question[idx])
        #     idx += 1
        #     continue
        if dataset_name == "sparc":
            tok = " ".join(tok.split("\ufeff")).strip()
        for mark in change_marks:
            tok = tok.replace(mark, "\"")
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += [tok[0], tok[1:-1], tok[-1]]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += [tok[0], tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], tok[-1]]
        elif tok in quotation_marks:
            new_question.append(tok)
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += [tok[0], tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
        idx += 1
    return new_question

class Preprocessor():

    def __init__(self, dataset_name, db_dir='data/database', db_content=True):
        super(Preprocessor, self).__init__()
        self.db_dir = db_dir
        self.db_content = db_content
        self.nlp_tokenize = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized = False, use_gpu=True)#, use_gpu=False)
        self.nlp_pretokenize = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized = True, use_gpu=True)#, use_gpu=False)
        self.stopwords = stopwords.words("english")

    def pipeline(self, entry: dict, db: dict, dataset_name: str, data_idx: int):
        entry["final_preprocessed_text_list"] = []
        if dataset_name in ["cosql", "sparc"]:
            entry =  self.multi_turn_pipeline(entry, db, dataset_name, data_idx)
        else:
            entry = self.single_turn_pipeline(entry, db, dataset_name, data_idx)
        return entry

    def multi_turn_pipeline(self, entry: dict, db: dict, dataset_name: str, data_idx: int):
        """ db should be preprocessed """
        # if coref_dataset is not None:
        #     entry['coref'] = coref_dataset[data_idx]["coref"]
        #     entry['coref_text_list'] = coref_dataset[data_idx]["text_list"]
        entry["text_list"] = []
        question = entry['final']['utterance']
        entry["text_list"].append(question)
        entry["text_list"].extend([" ".join(quote_normalization(dataset_name, data_idx, entry["interaction"][turn]["utterance_toks"])) for turn in range(len(entry['interaction']))])
        entry["processed_text_list"] = []
        for q in entry["text_list"]:
            q = q.split("|")
            entry["processed_text_list"].append(q)
        # print(entry["processed_text_list"])
        for idx, turn_item in enumerate(entry['processed_text_list']):
            idx = str(idx)
            if len(turn_item) > 1:
                for i in range(len(turn_item)):
                    multi_idx = idx + "#"+str(i)
                    if i== 0 :
                        entry[multi_idx] = turn_item[1].count("|")+1
                    entry = self.preprocess_question(entry, db, multi_idx, dataset_name, data_idx)
                    entry = self.schema_linking(entry, db,  multi_idx)
            else:
                entry = self.preprocess_question(entry, db,  idx, dataset_name, data_idx)
                entry = self.schema_linking(entry, db,  idx)  
        return entry

    def single_turn_pipeline(self, entry: dict, db: dict, dataset_name: str, data_idx: int):
        entry = self.preprocess_question(entry, db, "-1", dataset_name, data_idx)
        entry = self.schema_linking(entry, db, "-1")
        return entry

    def preprocess_database(self, db: dict):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks, table_names = [], []
        for tab in db['table_names']:
            doc = self.nlp_tokenize(tab)
            tab = [w.lemma.lower() for s in doc.sentences for w in s.words]
            table_toks.append(tab)
            table_names.append(" ".join(tab))
        db['processed_table_toks'], db['processed_table_names'] = table_toks, table_names
        column_toks, column_names = [], []
        for _, c in db['column_names']:
            doc = self.nlp_tokenize(c)
            c = [w.lemma.lower() for s in doc.sentences for w in s.words]
            column_toks.append(c)
            column_names.append(" ".join(c))
        db['processed_column_toks'], db['processed_column_names'] = column_toks, column_names
        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [[] for _ in range(len(table_names))] # from table id to column ids list
        for col_id, col in enumerate(db['column_names']):
            if col_id == 0: continue
            table2columns[col[0]].append(col_id)
        db['column2table'], db['table2columns'] = column2table, table2columns

        t_num, c_num, dtype = len(db['table_names']), len(db['column_names']), '<U100'

        # relations in tables, tab_num * tab_num
        tab_mat = np.array([['table-table-generic'] * t_num for _ in range(t_num)], dtype=dtype)
        table_fks = set(map(lambda pair: (column2table[pair[0]], column2table[pair[1]]), db['foreign_keys']))
        for (tab1, tab2) in table_fks:
            if (tab2, tab1) in table_fks:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fkb', 'table-table-fkb'
            else:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fk', 'table-table-fkr'
        tab_mat[list(range(t_num)), list(range(t_num))] = 'table-table-identity'

        # relations in columns, c_num * c_num
        col_mat = np.array([['column-column-generic'] * c_num for _ in range(c_num)], dtype=dtype)
        for i in range(t_num):
            col_ids = [idx for idx, t in enumerate(column2table) if t == i]
            col1, col2 = list(zip(*list(product(col_ids, col_ids))))
            col_mat[col1, col2] = 'column-column-sametable'
        col_mat[list(range(c_num)), list(range(c_num))] = 'column-column-identity'
        if len(db['foreign_keys']) > 0:
            col1, col2 = list(zip(*db['foreign_keys']))
            col_mat[col1, col2], col_mat[col2, col1] = 'column-column-fk', 'column-column-fkr'
        col_mat[0, list(range(c_num))] = '*-column-generic'
        col_mat[list(range(c_num)), 0] = 'column-*-generic'
        col_mat[0, 0] = '*-*-identity'

        # relations between tables and columns, t_num*c_num and c_num*t_num
        tab_col_mat = np.array([['table-column-generic'] * c_num for _ in range(t_num)], dtype=dtype)
        col_tab_mat = np.array([['column-table-generic'] * t_num for _ in range(c_num)], dtype=dtype)
        cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), range(1, c_num))))) # ignore *
        col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-has', 'table-column-has'
        if len(db['primary_keys']) > 0:
            cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), db['primary_keys']))))
            col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-pk', 'table-column-pk'
        col_tab_mat[0, list(range(t_num))] = '*-table-generic'
        tab_col_mat[list(range(t_num)), 0] = 'table-*-generic'

        relations = np.concatenate([
            np.concatenate([tab_mat, tab_col_mat], axis=1),
            np.concatenate([col_tab_mat, col_mat], axis=1)
        ], axis=0)
        db['relations'] = relations.tolist()
        return db

    def read_tree(self, doc, tree_mat):
        trees = dict()
        root = None
        
        root_list = []
        
        bias = 0
        for sent in doc.sentences: 
            for word in sent.words:
                tree = Tree()
                tree.idx = word.id -1 + bias 
                trees[tree.idx] = tree
            bias += len(sent.words)
        bias = 0
        for idx, sent in enumerate(doc.sentences): 
            # trees = trees_list[idx]
            for word in sent.words:
                head_id = word.head - 1 + bias
                word_id = word.id - 1 + bias
                if word.head - 1 == -1:
                    root = trees[word_id]
                    root_list.append([root, bias+len(sent.words)])
                    continue
                # tree_mat[head_id, word.id-1] = word.deprel
                trees[head_id].add_child(trees[word_id])
                tree_mat[head_id, word_id] = "Forward-Syntax"
                tree_mat[word_id, head_id] = "Backward-Syntax"
            bias += len(sent.words)
        return root_list, tree_mat.tolist()

    def seperate_sent(self, text):
        seperate_pattern = r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!)\"*\s*\s*(?:\W*)([A-Z])"
        sep_text = re.sub(seperate_pattern, r"\n\1", text)
        return sep_text.split("\n")

    def preprocess_question(self, entry: dict, db: dict, turn: str, dataset_name: str, data_idx: int):
        """ Tokenize, lemmatize, lowercase question"""

        if turn == "-1":
            if dataset_name == "spider_dk" or dataset_name == "dk_spider_dates":
                question = entry["question"]
            elif dataset_name == "spider_syn":
                question = entry["SpiderSynQuestion"]
            else:
                question = " ".join(quote_normalization(dataset_name, data_idx, entry["question_toks"]))
            # question = "\n".join(self.seperate_sent(question))
            entry["processed_text_list"] = [question]
        elif "#" in turn:
            parent_idx, son_idx = turn.split("#")
            question = entry["processed_text_list"][int(parent_idx)][int(son_idx)]
            # question = "\n".join(self.seperate_sent(question))
        else:
            question = entry["processed_text_list"][int(turn)][0]
            # question = "\n".join(self.seperate_sent(question))
        question = question.strip()
        if turn == "0":
            doc = self.nlp_tokenize(question)
        else:
            doc = self.nlp_pretokenize(question)
        raw_toks = [w.text.lower() for s in doc.sentences for w in s.words]
        toks = [w.lemma.lower() if w.lemma else w.text.lower() for s in doc.sentences for w in s.words]
        entry[f'raw_question_toks_{turn}'] = raw_toks
        entry[f'ori_toks_{turn}'] = [w.text for s in doc.sentences for w in s.words]
        entry[f'processed_question_toks_{turn}'] = toks
        # print(question, [w.text for s in doc.sentences for w in s.words])
        entry["final_preprocessed_text_list"].append([turn, [w.text for s in doc.sentences for w in s.words], len([w.text for s in doc.sentences for w in s.words])])

        # relations in questions, q_num * q_num
        q_num, dtype = len(toks), '<U100'
        if q_num <= MAX_RELATIVE_DIST + 1:
            dist_vec = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity'
                for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)]
            starting = MAX_RELATIVE_DIST
        else:
            dist_vec = ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1) + \
                ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' \
                    for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
                    ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1)
            starting = q_num - 1
        q_mat = np.array([dist_vec[starting - i: starting - i + q_num] for i in range(q_num)], dtype=dtype)
        entry[f'relations_{turn}'] = q_mat.tolist()

        tree_mat = np.array([["None-Syntax"] * q_num for _ in range(q_num)], dtype=dtype)
        root_list, tree_mat = self.read_tree(doc, tree_mat)
        entry[f'tree_relations_{turn}'] = tree_mat
        return entry


    def schema_linking(self, entry: dict, db: dict, turn: str):
        """ Perform schema linking: both question and database need to be preprocessed """
        raw_question_toks, question_toks = entry[f'raw_question_toks_{turn}'], entry[f'processed_question_toks_{turn}']
        table_toks, column_toks = db['processed_table_toks'], db['processed_column_toks']
        table_names, column_names = db['processed_table_names'], db['processed_column_names']
        q_num, t_num, c_num, dtype = len(question_toks), len(table_toks), len(column_toks), '<U100'

        # relations between questions and tables, q_num*t_num and t_num*q_num
        table_matched_pairs = {'partial': [], 'exact': []}
        q_tab_mat = np.array([['question-table-nomatch'] * t_num for _ in range(q_num)], dtype=dtype)
        tab_q_mat = np.array([['table-question-nomatch'] * q_num for _ in range(t_num)], dtype=dtype)
        max_len = max([len(t) for t in table_toks])
        index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
        for i, j in index_pairs:
            phrase = ' '.join(question_toks[i: j])
            if phrase in self.stopwords: continue
            for idx, name in enumerate(table_names):
                if phrase == name: # fully match will overwrite partial match due to sort
                    q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
                    tab_q_mat[idx, range(i, j)] = 'table-question-exactmatch'
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'
                    tab_q_mat[idx, range(i, j)] = 'table-question-partialmatch'

        # relations between questions and columns
        column_matched_pairs = {'partial': [], 'exact': [], 'value': []}
        q_col_mat = np.array([['question-column-nomatch'] * c_num for _ in range(q_num)], dtype=dtype)
        col_q_mat = np.array([['column-question-nomatch'] * q_num for _ in range(c_num)], dtype=dtype)
        max_len = max([len(c) for c in column_toks])
        index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
        for i, j in index_pairs:
            phrase = ' '.join(question_toks[i: j])
            if phrase in self.stopwords: continue
            for idx, name in enumerate(column_names):
                if phrase == name: # fully match will overwrite partial match due to sort
                    q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-exactmatch'
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_col_mat[range(i, j), idx] = 'question-column-partialmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-partialmatch'
        if self.db_content:
            db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
            if not os.path.exists(db_file):
                raise ValueError('[ERROR]: database file %s not found ...' % (db_file))
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors='ignore')
            conn.execute('pragma foreign_keys=ON')
            for i, (tab_id, col_name) in enumerate(db['column_names_original']):
                if i == 0 or 'id' in column_toks[i]: # ignore * and special token 'id'
                    continue
                tab_name = db['table_names_original'][tab_id]
                try:
                    cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (col_name, tab_name))
                    cell_values = cursor.fetchall()
                    cell_values = [str(each[0]) for each in cell_values]
                    cell_values = [[str(float(each))] if is_number(each) else each.lower().split() for each in cell_values]
                except Exception as e:
                    print(e)
                for j, word in enumerate(raw_question_toks):
                    word = str(float(word)) if is_number(word) else word
                    for c in cell_values:
                        if word in c and 'nomatch' in q_col_mat[j, i] and word not in self.stopwords:
                            q_col_mat[j, i] = 'question-column-valuematch'
                            col_q_mat[i, j] = 'column-question-valuematch'
                            break
            conn.close()

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
        q_col_mat[:, 0] = 'question-*-generic'
        col_q_mat[0] = '*-question-generic'
        q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
        entry[f'schema_linking_{turn}'] = (q_schema.tolist(), schema_q.tolist())
        return entry
