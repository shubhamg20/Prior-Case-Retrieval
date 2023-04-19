from transformers import AutoTokenizer 
import threading
import re, string, json
import pandas as pd
import numpy as np
import pickle
def regex(file_path):
    with open( file_path, 'r') as  file_path:
        print("Here")
        raw_text =  file_path.read()
        file_path.close()
        raw_text = re.sub(r"\xa0"," ",raw_text)
        pattern = r'[a-zA-Z]+\.[a-zA-Z]+\.?'
        # replace the matched words with an empty string
        raw_text = re.sub(pattern, '', raw_text)
        pattern = r'-\s'
        # replace the matched pattern with an empty string
        raw_text = re.sub(pattern, '', raw_text)
        raw_text = raw_text.lower()
        raw_text = raw_text.split("\n") # splitting using new line character
        text = raw_text.copy()
        text = [re.sub(r'[^a-zA-Z0-9.:&,[\]<_>)\-(/?\t ]','',sentence) for sentence in text]
        #text = [re.sub(r'[^a-zA-Z0-9.,)\-(/?\t ]','',sentence) for sentence in text]
        # text1 = [re.sub(r'(?<=[^0-9])/(?=[^0-9])',' ',sentence) for sentence in text1]
        text = [re.sub("\t+"," ",sentence) for sentence in text]
        text = [re.sub("\s+"," ",sentence) for sentence in text] # converting multiple tabs and spaces ito a single tab or space
        text = [re.sub(" +"," ",sentence) for sentence in text]
        text = [re.sub("\.\.+","",sentence) for sentence in text] # these were the commmon noises in out data, depends on data
        text = [re.sub("\A ?","",sentence) for sentence in text]
        text = [sentence for sentence in text if(len(sentence) != 1 and not re.fullmatch("(\d|\d\d|\d\d\d)",sentence))]
        text = [sentence for sentence in text if len(sentence) != 0]
        text = [re.sub('\A\(?(\d|\d\d\d|\d\d|[a-zA-Z])(\.|\))\s?(?=[A-Z])','\n',sentence) for sentence in text]#dividing into para wrt to points
        text = [re.sub("\A\(([ivx]+)\)\s?(?=[a-zA-Z0-9])",'\n',sentence) for sentence in text] #dividing into para wrt to roman points
        #text = [re.sub(r"[()[\]\"$']"," ",sentence) for sentence in text]
        text = [re.sub(r" no\."," number ",sentence,flags=re.I) for sentence in text]
        text = [re.sub(r" &"," and ",sentence,flags=re.I) for sentence in text]
        text = [re.sub(r" was\."," was ",sentence,flags=re.I) for sentence in text]
        text = [re.sub(r" rs\."," rupees ",sentence,flags=re.I) for sentence in text]
        text = [re.sub(r" vs\."," vs ",sentence,flags=re.I) for sentence in text]
        text = [re.sub(r" nos\."," numbers ",sentence,flags=re.I) for sentence in text]
        text = [re.sub(r" co\."," company ",sentence) for sentence in text]
        text = [re.sub(r" ltd\."," limited ",sentence,flags=re.I) for sentence in text]
        text = [re.sub(r" pvt\."," private ",sentence,flags=re.I) for sentence in text]
        # text = [re.sub("\s+"," ",sentence) for sentence in text]
        text2 = []
        for index in range(len(text)):#for removing multiple new-lines
            if(index>0 and text[index]=='' and text[index-1]==''):
                continue
            if(index<len(text)-1 and text[index+1]!='' and text[index+1][0]=='\n' and text[index]==''):
                continue
            text2.append(text[index])
        text = text2
        alphabet_string = string.ascii_lowercase
        alphabet_list = list(alphabet_string)
        exclude_list = alphabet_list + [
                            "sub-s",
                            "supl",
                            "subs",
                            "ss",
                            "cl",
                            "dr",
                            "mr",
                            "mrs",
                            "ms",
                            "vs",
                            "ch",
                            "addl",]
        exclude_list = [word + "." for word in exclude_list]
        text = "\n".join(text)
        text = [word for word in text.split() if word.lower() not in exclude_list]
        text = " ".join(text)
        lines = text.split("\n")
        text_new = " ".join(lines)
        text_new = re.sub(" +"," ",text_new)
        return text_new

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

def process_query(query, data, output_file):
    match = re.search(r'0*(\d+)\.txt', query)
    query_text = data['dict_query'][int(match.group(1))]
    query_text = " ".join(query_text)
    query_list = doc_split(query_text)

    candidates = csv.loc[csv['Query Case'] == query, 'Cited Cases'].iloc[0].split(',')
    cand_list = []
    for candidate in candidates:
        string_ori = candidate
        candidate = re.sub(r'[^a-zA-Z0-9.]', '', candidate)
        match = re.search(r'0*(\d+)\.txt', candidate)
        if(match):
            cand_text = data['dict_candidate'][int(match.group(1))]
            cand_text = " ".join(cand_text)
            cand_list.extend(doc_split(cand_text))
        else:
            print("------"+string_ori+"-------")

    dict_list = []
    for j, query in enumerate(query_list):
        term_recall_dict = {}
        json_dict = {
            "query": query,
            "qid": j,
            "doc": cand_list,
            "term_recall": term_recall_dict
        }
        
        dict_list.append(json_dict)

    with open(output_file, "a") as f:
        for dict_ in dict_list:
            out_str = json.dumps(dict_)
            f.write(out_str + '\n')
if __name__ == '__main__':
    with open('event_text_dict_test.sav', 'rb') as f:
        data=pickle.load(f)
        f.close()
        # Load data and CSV file
        csv = pd.read_csv('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test/test.csv')

        # Create threads to process queries
        threads = []
        output_file = "/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/data_test.txt"
        for query in csv['Query Case']:
            thread = threading.Thread(target=process_query, args=(query, data, output_file))
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        
            
    