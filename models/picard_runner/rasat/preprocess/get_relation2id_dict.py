

def get_relation2id_dict(choice = "Default", use_coref = False, use_dependency = False):
    from .constants import RELATIONS, MAX_RELATIVE_DIST
    current_relation = [r for r in RELATIONS]
    if not use_coref:
        current_relation = [r for r in current_relation if r not in ['co_relations', 'coref_relations']]
    if not use_dependency:
        current_relation = [r for r in current_relation if r not in ['Forward-Syntax', 'Backward-Syntax', 'None-Syntax']]
    if choice in ["Default"]:
        idx_list = [i for i in range(1, len(current_relation)+1)]
    elif choice == "DefaultWithoutSchemaEncoding":
        schema_encoding_rel = []
        for rel in current_relation:
            split_rel = rel.split("-")
            try:
                src_type, tgt_type = split_rel[0], split_rel[1]
            except:
                continue
            if src_type in ["table", "column", "*"] and tgt_type in ["table", "column", "*"]:
                schema_encoding_rel.append(rel)
        current_relation = [r for r in current_relation if r not in schema_encoding_rel]
        idx_list = [i for i in range(1, len(current_relation)+1)]
        for rel in schema_encoding_rel:
            current_relation.append(rel)
            idx_list.append(0)
    elif choice == "DefaultWithoutSchemaLinking":
        schema_linking_rel = []
        for rel in current_relation:
            split_rel = rel.split("-")
            try:
                src_type, tgt_type = split_rel[0], split_rel[1]
            except:
                continue
            if (src_type in ["question"] and tgt_type in ["table", "column", "*"]) or (tgt_type in ["question"] and src_type in ["table", "column", "*"]):
                schema_linking_rel.append(rel)
        current_relation = [r for r in current_relation if r not in schema_linking_rel]
        idx_list = [i for i in range(1, len(current_relation)+1)]
        for rel in schema_linking_rel:
            current_relation.append(rel)
            idx_list.append(0)
    elif choice == "MinType":
        idx_list = []
        dummy_idx = 8
        for rel in current_relation:
            if rel in ['question-column-partialmatch', 'question-table-partialmatch']:
                idx_list.append(1)
            elif rel in ['question-column-exactmatch', 'question-table-exactmatch']:
                idx_list.append(2)
            elif rel in ['question-column-valuematch']:
                idx_list.append(3)
            elif rel in ['question-table-nomatch', 'question-column-nomatch']:
                idx_list.append(4)
            elif rel in ['table-column-pk']:
                idx_list.append(5)
            elif rel in ['table-column-has']:
                idx_list.append(6)
            elif rel in ['column-column-fk']:
                idx_list.append(7)
            elif rel in ['question-question-generic'] + ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)]:
                idx_list.append(dummy_idx)
                dummy_idx += 1
            else:
                idx_list.append(0)
    elif choice == "Dependency_MinType":
        idx_list = []
        dummy_idx = 8
        for rel in current_relation:
            if rel in ['question-column-partialmatch', 'question-table-partialmatch']:
                idx_list.append(1)
            elif rel in ['question-column-exactmatch', 'question-table-exactmatch']:
                idx_list.append(2)
            elif rel in ['question-column-valuematch']:
                idx_list.append(3)
            elif rel in ['question-table-nomatch', 'question-column-nomatch']:
                idx_list.append(4)
            elif rel in ['table-column-pk']:
                idx_list.append(5)
            elif rel in ['table-column-has']:
                idx_list.append(6)
            elif rel in ['column-column-fk']:
                idx_list.append(7)
            elif rel in ['Forward-Syntax', 'Backward-Syntax', 'None-Syntax']:
                idx_list.append(dummy_idx)
                dummy_idx += 1
            else:
                idx_list.append(0)
    else:
        raise NotImplementedError
    RELATION2ID_DICT = dict(zip(current_relation, idx_list))
    idx_list.append(0)
    current_relation.append("None")
    ID2RELATION_DICT = dict(zip(idx_list, current_relation))
    return RELATION2ID_DICT, ID2RELATION_DICT, max(idx_list)