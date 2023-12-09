
import spacy
from spacy.tokens import Doc
import coreferee
import json
from tqdm import tqdm
import os

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)

def init_nlp():
    nlp = spacy.load('en_core_web_trf')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    nlp.add_pipe('coreferee')
    return nlp

def find_turn_idx(id, turn_list, len_text_list):
    total_length = 0
    for idx, length in enumerate(len_text_list):
        total_length += length
        if id < total_length and id >= total_length-length:
            return turn_list[idx], id-total_length+length
    return turn_list[0], id

def text_list2coref_json(output_path, mode, nlp):
    with open(os.path.join(output_path, f"{mode}_text_list.txt"), 'r') as load_f: 
        dataset = load_f.readlines()
    new_res=[]
    for idx, entry in tqdm(enumerate(dataset)):
        final_preprocessed_text_list = eval(dataset[idx].strip())
        text_list = " ".join([i for item in final_preprocessed_text_list for i in item[1]])
        turn_list = [item[0] for item in final_preprocessed_text_list]
        len_text_list = [item[2] for item in final_preprocessed_text_list]
        
        doc = nlp(text_list)
        coref_dict = {}
        for chain in doc._.coref_chains:
            key = chain.index
            used_turn = set()
            coref_dict[key] = {}
            coref_dict[key]["group"] = []
            for li in [list(_) for _ in chain]:
                new_list = []
                for idx in li:
                    item = find_turn_idx(idx, turn_list, len_text_list)
                    item_dict = {"turn": item[0], "position": item[1], "ori": idx}
                    used_turn.add(item[0])
                new_list.append(item_dict)
                coref_dict[key]["group"].append(new_list)
            coref_dict[key]["used_turn"] = list(used_turn)
        new_entry = {}
        new_entry["coref"] = coref_dict
        new_entry["text_list"] = text_list
        new_res.append(new_entry)
    with open(os.path.join(output_path, f'{mode}_coref.json'),"w") as dump_f:
        json.dump(new_res,dump_f) 

def main():
    mode_list = ["train", "dev"]
    dataset_name_list = ["spider", "cosql", "spider_dk"]
    nlp = init_nlp()
    for datatset_name in dataset_name_list:
        for mode in mode_list:
            if datatset_name == 'spider_dk' and mode == 'train':
                continue
            output_path = os.path.join("../../dataset_files/preprocessed_dataset/", datatset_name)
            text_list2coref_json(output_path, mode, nlp)

main()
