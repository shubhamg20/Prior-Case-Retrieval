import csv
import os, codecs
import pickle
import pandas as pd
import re
import tqdm
with open('event_text_dict_test.sav', 'rb') as f:
    data=pickle.load(f)
    f.close()
        

for key in data['dict_query'].keys():
    list_of_sent = data['dict_query'][key]
    file_name = str(key).zfill(10)
    with open('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/query_pool_test/'+file_name+'.txt','w') as f1:
        out_str = ' '.join(list_of_sent)
        print(file_name)
        f1.write(out_str)
# gold_labels_csv = '/workspace/tejas/PCR/DeepCT/scripts/bm25_code/bm25_code/data2.csv'
# gold_labels = pd.read_csv(gold_labels_csv)
# for i in gold_labels.columns.values[0:]:
#     print(i)
#     print(type(i))
#candidate_docs = [int(i) for i in gold_labels.columns.values[1:]]