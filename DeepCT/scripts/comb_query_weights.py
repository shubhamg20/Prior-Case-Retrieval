import threading
import re, string, json, csv
import pandas as pd
import numpy as np
import pickle

query_dict = {}

with open('/workspace/tejas/PCR/DeepCT/scripts/deepctw_04april.json') as json_file:
    dict_req = json.load(json_file)

for line1 in open("/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_index.txt"):
    json_dict1 = json.loads(line1)
    id_list = json_dict1['id']
    file_name = json_dict1['query_id']
    term_weights = {}
    for id in id_list:
        term_weights_dict = dict_req[str(id)]
        for token in term_weights_dict.keys():
            term_weights[token] = max(term_weights_dict[token],term_weights.get(token, 0))
    query_dict[file_name] = term_weights
    print(len(term_weights.keys()))

with open('comb_query_weights_04april.json','w') as f:
    json.dump(query_dict,f)
            
                
        
    
    



    