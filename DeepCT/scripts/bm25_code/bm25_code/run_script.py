import os, sys, json, time, numpy as np
import re
from evaluate_at_K_new import *
from sklearn.model_selection import ParameterGrid

#assert(len(sys.argv) == 2)

current_time = '_'.join(time.ctime().split()) + '_' + str(np.random.randint(0,1000))
save_folder = f'./exp_results/EXP_{current_time}'
os.makedirs(save_folder)

# with open(sys.argv[1], 'r') as f:
#     config_dict = json.load(f)
# with open(save_folder + f'/config_file.json', 'w') as f:
#     json.dump(config_dict, f, indent = 4)

config_dict={
    'path_prior_cases' : '/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/cand_pool_test',
    'path_current_cases' : '/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/query_pool_test',
    'true_labels_csv' : '/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test/test.csv',
    'n_gram' : 1,
    'sim_csv_path' : '/workspace/tejas/PCR/DeepCT/scripts/bm25_code/bm25_code/data2.csv',
    'gold_label_file_path' : '/workspace/tejas/PCR/DeepCT/scripts/bm25_code/bm25_code/gold_data.csv'
}

path_prior_cases = config_dict['path_prior_cases']
path_current_cases = config_dict['path_current_cases']
true_labels_csv = config_dict['true_labels_csv']
n_gram = config_dict['n_gram']
sim_csv_path = config_dict['sim_csv_path']
gold_label_file_path = config_dict['gold_label_file_path']
assert(os.path.isdir(path_prior_cases))
assert(os.path.isdir(path_current_cases))
assert(os.path.isfile(true_labels_csv))
assert(os.path.isfile(sim_csv_path))
# assert(os.path.isfile(gold_label_file_path))

bm25_results_save_dict_path = f'{save_folder}/bm25_results.sav'
filled_sim_csv_path = f'{save_folder}/filled_similarity_matrix.csv'

# print(
#     path_prior_cases, 
#     path_current_cases,
#     true_labels_csv,
#     n_gram,
#     sim_csv_path,
#     gold_label_file_path,
#     bm25_results_save_dict_path,
#     filled_sim_csv_path
# )

import os
import codecs
# import spacy
# from spacy.attrs import ORTH
import re
import string

# import nltk
# from nltk.stem import PorterStemmer 
# from nltk.tokenize import word_tokenize 
# import nltk.data

import time
import random
# import multiprocessing as mp

import tqdm
import pandas as pd
import csv 
from csv import reader
# import ast
import pickle as pkl

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

class BM25(object):
    def __init__(self, b=0.7, k1=1.6, n_gram:int = 1):
#         self.vectorizer = c(tokenizer=stemming_tokenizer, 
#                                           max_df=.90, min_df=1,
#                                           stop_words='english', 
#                                           use_idf=True, 
#                                           ngram_range=(2, 2))
        self.n_gram = n_gram
        self.vectorizer = TfidfVectorizer(max_df=.65, min_df=1,
                                  use_idf=True, 
                                  ngram_range=(n_gram, n_gram))
        
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        start_time = time.perf_counter()
        print(f"Fitting tf_idf vectorizer")
        self.vectorizer.fit(X)
        print(f"Finished tf_idf vectorizer, time : {time.perf_counter() - start_time:0.3f} sec")
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1

my_suffixes = (".txt")
citation_file_paths = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path_prior_cases):
#     print(r,len(r))
    for file in f:
#         print(file)
        if file.endswith(my_suffixes):
            citation_file_paths.append(os.path.join(r, file))
            
# '/workspace/tejas/PCR/DeepCT/scripts/bm25_code/bm25_code/exp_results/EXP_Mon_Mar_20_16:07:21_2023_509/filled_similarity_matrix.csv'
name_dict = {}
corpus =[]
citation_names = []
for file in sorted(citation_file_paths):
#     print(file)
    f = codecs.open(file, "r", "utf-8", errors='ignore')
    text = f.read()
    corpus.append(text)
    citation_names.append(os.path.basename(file))
    name_dict[text] = os.path.basename(file)


my_suffixes = (".txt")
query_file_paths = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path_current_cases):
#     print(r,len(r))
    for file in f:
#         print(file)
        if file.endswith(my_suffixes):
            query_file_paths.append(os.path.join(r, file))


query_corpus = []
query_names = [] 


#iterate throught the query database list in sorted manner
for file in tqdm.tqdm(sorted(query_file_paths),desc = "query documents"):
    open_file = open(file, 'r', encoding="utf-8")
    text = open_file.read()
    raw_str_list = text.splitlines()
    str_list_3 = raw_str_list
    query_corpus.append(''.join(str_list_3))
    query_names.append(os.path.basename(file).zfill(14))
    open_file.close()

#STORE ACTUAL NUMBER OF CITATIONS IN DICTIONARY
golden_citations = {}
golden = {}

df = pd.read_csv(true_labels_csv)
for i in range(df.shape[0]):
    query_case = df.iloc[i]['Query Case']
    # query_case = query_case
    query_case = query_case.zfill(14)
    candidate_cases = df.iloc[i]['Cited Cases']
    candidate_cases = [i for i in re.findall(r'\d+.txt', candidate_cases)]
    candidate_cases = [re.search(r'0*(\d+)\.txt', candidate) for candidate in candidate_cases]
    golden[query_case] = len(candidate_cases)
    golden_citations[query_case] = candidate_cases

from pydoc import doc
    
param_grid = {
    'k1': [2.0],      #2.0
    'b': [1.2]       #1.2
}
results = []
param_combinations = ParameterGrid(param_grid)
for params in param_combinations:
    print("Testing on k1=", params['k1'], " b=", params['b'])
    bm25 = BM25(params['b'], params['k1'], n_gram = n_gram)
    bm25.fit(corpus)
    score = 0.0
    score_dict = {}
    prediction_dict = {}
    pred_df = pd.DataFrame(columns=['Document id','No of Golden Citations','Min BM25 Sim Value in TOP R','Actual Citations','Prediction List'])
    bm_25_results_dict = {}

    for i in tqdm.tqdm(range(len(query_corpus))):
        with open(save_folder + f'/progress.txt', 'a+') as logger:
            logger.write(f'Doing {i}/{len(query_corpus)}\n')

        # i = query_names.index('0000021652.txt')
        qu = query_corpus[i]
        qu_n = query_names[i]
        R = golden[qu_n]
        doc_scores = bm25.transform(qu, corpus)
        
        assert(int(re.findall(r'\d+',qu_n)[0]) not in bm_25_results_dict)

        bm_25_results_dict[int(re.findall(r'\d+',qu_n)[0])] = {int(re.findall(r'\d+',citation_names[i])[0]) : doc_scores[i] for i in range(len(doc_scores))}

        # print(bm_25_results_dict, file = open(f'./stupid_logs/{i}.txt', 'w+'))

        # if len(bm_25_results_dict.keys()) == 1:
        #     print(bm_25_results_dict)
        #     raise RuntimeError

    with open(bm25_results_save_dict_path, 'wb') as f:
        pkl.dump(bm_25_results_dict, f)

    sim_df = pd.read_csv(sim_csv_path)
    del_columns = [i for i in sim_df.columns if i.startswith('Unnamed')]
    sim_df = sim_df.drop(columns=del_columns, axis = 1)
    column_candidates = list(sim_df.columns)[1:]
    column_name = 'query_case_id' if 'query_case_id' in sim_df.columns else 'Unnamed: 0'
    for i, query in tqdm.tqdm(enumerate(list(sim_df[column_name].values))):
        assert(sim_df.iloc[i][column_name] == query)
        n=30721
        if query not in bm_25_results_dict.keys():
            print('bsdk')
        temp_bm25_scores = [bm_25_results_dict[query][int(i)] for i in column_candidates]
        sim_df.iloc[i] = [float(query)] + temp_bm25_scores

    sim_df.to_csv(filled_sim_csv_path)

    # do this
    output_numbers = get_f1_vs_K(gold_label_file_path, filled_sim_csv_path)
    print("F1: ", max(output_numbers['f1_vs_K']), "Params:", params)
    results.append((params, output_numbers))
best_f1_k_list=[]
for res in results:    
    best_f1_k_list.append((res[0], max(res[1]['f1_vs_K'])))
best_params, best_score = max(best_f1_k_list, key=lambda x: x[1])
print(f"Best hyperparameters: {best_params}")
print(f"Best score: {best_score}")

for res in results:
    if(int(res[0]['k1']==best_params['k1']) & int(res[0]['b']==best_params['b'])):
        output=res[1]
with open(f'{save_folder}/output_final_{best_params["k1"]}_{best_params["b"]}_{config_dict["n_gram"]}.json', 'w') as f:
    json.dump(output, f, indent = 4)

# save in new folder, name marked with time -> Done
# # also save copy of config file in folder  -> Done
