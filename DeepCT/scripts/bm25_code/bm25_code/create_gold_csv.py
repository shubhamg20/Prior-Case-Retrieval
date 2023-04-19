import csv
import os, codecs
import pandas as pd
import re
import tqdm
def clean_cand(cand):
    return str(int(cand.split('.')[0]))

path_current_cases = '/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_reweighted'
query_file_paths = []
my_suffixes = 'txt'
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

print(citation_names)

df = pd.DataFrame(columns = citation_names)
initial_values = {'query_case_id': query_names}
for col in df.columns[1:]:
    initial_values[col] = 0.0
df = pd.DataFrame(initial_values)

test_data = pd.read_csv('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test/test.csv')
test_df = pd.DataFrame(data = test_data)
print(df.columns[1])
test_df['query_case_id'] = test_df['Query Case'].apply(clean_cand)
counter = 0
for query in test_df['query_case_id']:
    print(counter)
    candidates = test_df.loc[test_df['query_case_id'] == query, 'Cited Cases'].iloc[0].split(',')
    for candidate in candidates:
        org_string = candidate
        candidate = re.sub(r'[^a-zA-Z0-9.]', '', candidate)
        match = re.search(r'0*(\d+)\.txt', candidate)
        if(match):
            print(org_string)
            can = str(int(match.group(1)))
            if can in df.columns:
                print("There")
                df.loc[df['query_case_id'] == query,can] = 1
            else:
                print(f'{query} and {can} and {org_string} correspondence\n')        
    counter = counter + 1
df.to_csv('gold_data.csv')