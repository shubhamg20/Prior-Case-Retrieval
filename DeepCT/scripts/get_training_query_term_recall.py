import argparse
import json
import re
import string
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.data.path.append("/home/era/nltk_data")


stopwords = set([line.strip() for line in open("../data/stopwords.txt")]) 
stemmer = PorterStemmer()

def text_clean(text, stem, stop):
    text = text.replace("\'s", " ")
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text) 
    if stop:
       tokens = [t for t in tokens if t.lower() not in stopwords]
    if stem:
        new_tokens = [stemmer.stem(t.lower()) for t in tokens]
    else:
        new_tokens = [t.lower() for t in tokens]
    return ' '.join(new_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json_in_file", help="Each line: {\"queries\": [\"what kind of animals are in grasslands\", \"tropical grasslands animals\", ...], \"doc\":{\"title\": Tropical grassland animals (which do not all occur in the same area) include giraffes, zebras, buffaloes, ka...}}")
    parser.add_argument("--stem", action="store_true", help="recommend: true")
    parser.add_argument("--stop", action="store_true", help="recommend: true")
    args = parser.parse_args()
    dict_list=[]
    for line in open(args.json_in_file):
        json_dict = json.loads(line)
        json_dict["term_recall"] = {} 
        cands = json_dict["doc"]
        ctokens = {}
        for cand in cands:
            ctext = text_clean(cand, args.stem, args.stop)
            if not ctext: 
                continue
            tokens = set(ctext.split(' '))
            for t in tokens:
                ctokens[t] = ctokens.get(t, 0.0) + 1.0/len(cands)

        q_text = json_dict['query']
        q_text = text_clean(q_text, False, args.stop)
        q_tokens = set(q_text.split(' '))
        q_term_recall = {}
        for ttoken in q_tokens:
            ttoken2 = ttoken
            if args.stem: ttoken2 = stemmer.stem(ttoken)
            if ttoken2 in ctokens:
                q_term_recall[ttoken] = ctokens[ttoken2]
        json_dict["term_recall"] = q_term_recall 
        dict_list.append(json_dict)
        out_str = json.dumps(json_dict)
        print(out_str)
    with open("/home/era/PCR/DeepCT/data/PCR_UCREAT/data_recall.txt", "a") as f:
        for dict_ in dict_list:
            out_str = json.dumps(dict_)
            f.write(out_str + '\n')
    f.close()
    
