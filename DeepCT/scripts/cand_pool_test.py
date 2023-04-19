import pickle
with open('/workspace/tejas/PCR/DeepCT/scripts/bm25_code/bm25_code/exp_results/EXP_Sat_Mar_18_14:18:13_2023_748/bm25_results.sav', 'rb') as f:
    data=pickle.load(f)
f.close()
print(data[21652][1000149])
# for key in data['dict_candidate'].keys():
#     list_of_sent = data['dict_candidate'][key]
#     text = ' '.join(list_of_sent)
#     with open('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/cand_pool_test/'+str(key)+'.txt','w') as f:
#         f.write(text)
#     f.close()