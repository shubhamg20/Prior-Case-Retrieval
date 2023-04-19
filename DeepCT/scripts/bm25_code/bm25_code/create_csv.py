import csv
import os, codecs
import pandas as pd
import re
import tqdm

path_current_cases = '/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_reweighted'
query_file_paths = []
my_suffixes='txt'
for r, d, f in os.walk(path_current_cases):
    for file in f:
        if file.endswith(my_suffixes):
            query_file_paths.append(os.path.join(r, file))
query_names = [] # stores all the queries_
for file in tqdm.tqdm(sorted(query_file_paths),desc = "query documents"):
    query_names.append(os.path.basename(file).zfill(14))

query_names = [str(int(query.split('.')[0])) for query in query_names]
# list of candidate case names
path_prior_cases = '/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/cand_pool_test'
citation_file_paths = []

for r, d, f in os.walk(path_prior_cases):
    for file in f:
        if file.endswith(my_suffixes):
            citation_file_paths.append(os.path.join(r, file))
citation_names = ['query_case_id']

for file in sorted(citation_file_paths):
    cand = os.path.basename(file)
    cand = str(int(cand.split('.')[0]))
    citation_names.append(cand)

df = pd.DataFrame(columns = citation_names)
initial_values = {'query_case_id': query_names}
for col in df.columns[1:]:
    initial_values[col] = 0.0
df = pd.DataFrame(initial_values)
df.to_csv('data2.csv', index=False)