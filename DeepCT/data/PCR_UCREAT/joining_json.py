import threading
import re, string, json, csv
import pandas as pd
import numpy as np
import pickle

# for line in open("/workspace/tejas/PCR/DeepCT/data/ok_save_final.json", "r"):
#     for line1 in open("/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_index_new.txt"):
dict={}
for line in open("/workspace/tejas/PCR/DeepCT/data/ok_save_final.json", "r"):
    json_dict = json.loads(line)
    dict[int(json_dict['id'])] = json_dict['contents'] 

dict1 = {}
dict1["query_file_name"] = []
dict1["text"] = []
for line1 in open("/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_index_new.txt"):
    json_dict1 = json.loads(line1)
    id_list = json_dict1['id']
    file_name = json_dict1['query_id']
    list_of_sent = []
    for id in id_list:
        list_of_sent.append(dict[int(id)])
    
    out_str = ' '.join(list_of_sent)
    dict1["query_file_name"].append(file_name)
    dict1["text"].append(out_str)
    # with open('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_reweighted/'+file_name,'w') as f1:
    #     out_str = ' '.join(list_of_sent)
    #     print(file_name)
    #     f1.write(out_str)
print(dict1["query_file_name"])
with open('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_reweighted/my_dict.csv','w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(dict1.keys()) 
    for row in zip(*dict1.values()):
        writer.writerow(row)      
    #for line in open('/workspace/tejas/PCR/DeepCT/data/ok_save_final.json'):
        
        # json_dict = json.loads(line)
        # print(json_dict)
        # print('\n')
        # break
    