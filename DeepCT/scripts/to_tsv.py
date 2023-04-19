from transformers import AutoTokenizer 
import threading
import re, string, json, csv
import pandas as pd
import numpy as np
import pickle

def doc_split(document):
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    tokenized_document = tokenizer.encode(document)

    max_length = 512
    stride = 512
    # print(tokenized_document)
    # Split tokenized document into input segments of max_length tokens
    input_segments = []
    start = 0
    while start < len(tokenized_document):
        end = start + max_length
        if end >= len(tokenized_document):
            end = len(tokenized_document)
        input_segments.append(tokenizer.decode(tokenized_document[start:end]))
        start += stride
    return input_segments
    
if __name__ == '__main__':
    with open('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/event_text_dict_test.sav', 'rb') as f:
        data=pickle.load(f)
        f.close()
        # Load data and CSV file
        csv_file = pd.read_csv('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test/test.csv')

        # Create threads to process queries
        # output_file = "/home/era/PCR/DeepCT/data/PCR_UCREAT/test_data.tsv"
        count=0
        for query in csv_file['Query Case']:
            match = re.search(r'0*(\d+)\.txt', query)
            query_text = data['dict_query'][int(match.group(1))]
            query_text = " ".join(query_text)
            sentences = doc_split(query_text)
            
            with open('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_new.tsv', 'a', newline='', encoding='utf-8') as file:
                with open('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_index_new.txt', "a") as f:
                    writer = csv.writer(file, delimiter='\t')
                    dict={}
                    dict['query_id']= query
                    dict['id']=[]
                    for sentence in sentences: 
                        writer.writerow([count, sentence])
                        dict['id'].append(count)
                        count=count+1 
                    json_out=json.dumps(dict)
                    f.write(json_out + '\n')
                    

        