import pickle
import pandas as pd
with open('event_text_dict_train.sav', 'rb') as f:
    data=pickle.load(f)
    f.close()

print((data['dict_candidate'][599]))
# df=pd.DataFrame(data)   
# query=list(df['dict_query'])
# print(len(query))

# cand=list(df['dict_candidate'])
# print(len(cand))