import argparse
import json
import re
import string
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import threading
nltk.data.path.append("workspace/tejas/try/lib/python3.8/site-packages/nltk")


stopwords = set([line.strip() for line in open("../data/stopwords.txt")]) 
stemmer = PorterStemmer()
def process_line(json_dict,stem,stop):
    json_dict["term_recall"] = {} 
    cands = json_dict["doc"]
    ctokens = {}
    for cand in cands:
        ctext = text_clean(cand, stem, stop)
        if not ctext: 
            continue
        tokens = set(ctext.split(' '))
        for t in tokens:
            ctokens[t] = ctokens.get(t, 0.0) + 1.0/len(cands)

    q_text = json_dict['query']
    q_text = text_clean(q_text, False, stop)
    q_tokens = set(q_text.split(' '))
    q_term_recall = {}
    for ttoken in q_tokens:
        ttoken2 = ttoken
        if args.stem: ttoken2 = stemmer.stem(ttoken)
        if ttoken2 in ctokens:
            q_term_recall[ttoken] = ctokens[ttoken2]
    json_dict["term_recall"] = q_term_recall 
    out_str = json.dumps(json_dict)
    with open("/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/term_recall_test.txt", "a") as f:
        f.write(out_str + '\n')
        
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
    threads=[]
    for line in open(args.json_in_file):
        json_dict = json.loads(line)
        thread = threading.Thread(target=process_line, args=(json_dict,args.stem,args.stop))
        thread.start()
        threads.append(thread)
        # Wait for all threads to finish
    for thread in threads:
        thread.join()

##
# False positives

#Inference
# export BERT_BASE_DIR=/workspace/tejas/PCR/DeepCT/legal-bert-base-uncased
# export INIT_CKPT=/workspace/tejas/PCR/DeepCT/output_dir/model.ckpt-16338
# export TEST_DATA_FILE=/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_new.tsv
# export OUTPUT_DIR=/workspace/tejas/PCR/DeepCT/collections_04april
# /workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_new.tsv
# /workspace/tejas/PCR/DeepCT/collections/test_results.tsv
# /workspace/tejas/PCR/DeepCT/data/ok_save_final.json
# python run_deepct.py --task_name=marcotsvdoc --do_train=false --do_eval=false --do_predict=true --data_dir=$TEST_DATA_FILE --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/config.json --init_checkpoint=$INIT_CKPT --max_seq_length=512 --train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=3.0 --recall_field=title --output_dir=$OUTPUT_DIR


#Training
# export BERT_BASE_DIR=/workspace/tejas/PCR/DeepCT/legal-bert-base-uncased
# export INIT_CKPT=/workspace/tejas/PCR/DeepCT/legal-bert-base-uncased/legal-bert.ckpt
# export TRAIN_DATA_FILE=/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT
# export OUTPUT_DIR=/workspace/tejas/PCR/DeepCT/output_dir
# /workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_new.tsv
# /workspace/tejas/PCR/DeepCT/collections/test_results.tsv
# /workspace/tejas/PCR/DeepCT/data/ok_save_final.json
#python run_deepct.py --task_name=marcoquery --do_train=true --do_eval=false --do_predict=false --data_dir=$TRAIN_DATA_FILE --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/config.json --init_checkpoint=$INIT_CKPT --max_seq_length=512 --train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=5.0 --recall_field=title --output_dir=$OUTPUT_DIR


#python /workspace/tejas/tranformer/transformer/train.py -data_path=/workspace/tejas/tranformer/transformer/shubham-train.t7 -log=/workspace/tejas/tranformer/transformer/logs