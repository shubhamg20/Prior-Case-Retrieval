import json
import os.path
import pandas as pd
import ast
import scraper
folders= ["PCR_New",'PCR_New_4']
#folders= ["1",'2']

cit_folders=['Existing_Citations','Existing_Citations_4']
s={}
from IPython.display import display, HTML
map='map.csv'
all_ids='all_ids.csv'
all_cites='all_cites.csv'
all_cases='all_cases.csv'
import pandas as pd
from numpy.random import randint
def remove_csv_error(file):
    f = open(file, 'r', encoding='utf-8')
    data = f.readlines()
    data[0] = data[0][1::]
    with open(file, 'w', encoding='utf-8') as file:
        file.writelines(data)
def write_csv(new_data, file):
    try:
     df1 = pd.read_csv(file)
     #display(df1)
     df2=pd.DataFrame(new_data)
     #display(df2)
     df3 = pd.concat([df1, df2], ignore_index=True)
     df3.reset_index()
     df3.to_csv(file)#, index=[0])
     remove_csv_error(file)
    except:
      print('error')
      df = pd.DataFrame(new_data) #index=[0])
      df.to_csv(file)#, index=[0])
      remove_csv_error(file)



def addcase(filepath, id):
    #l=filepath.split('.')
    #filepath=l[0]+'2.csv'
    f = open(filepath, 'r')
    data = json.load(f)
    id=id.replace('.json','')
    a = []
    for i in data['all_citations']:
        a.append(int(i))
    cas=[]
    for x in data['doc_text']:
        x["paragraph_text"]=x["paragraph_text"].replace("\n", "")
        cas.append(x["paragraph_text"])
    d={'id': [id], 'case': [cas]}
    write_csv(d,'all_cases2.csv')
    return a
def add_all_cites(q,a):
    print(q,a)
    d={'id':[str(q)], 'cites': [a] }
    write_csv(d, all_cites)
def is_present(id, file=all_cites):
   try:
    df = pd.read_csv(file, index_col=[0])
    case_ids = df['id'].values.tolist()
    #cases = df['case'].values.tolist()
    if id in case_ids:
        return True
    return False
   except:
       return False
def exists(id):
    for folder in cit_folders:
        l=os.listdir(folder)
        for i in l:
            if str(id)==i.replace('.json',''):
                return True, folder
    else:
        return False, ''
def query_create():
    c = 0
    for folder in folders:
        query_cases = os.listdir(folder)
        for query in query_cases:
            query_no = int(query.replace('.json', ''))
            if not is_present(query_no):
                print('not present')
                filepath = os.path.join(folder, query)
                all_cites = addcase(filepath, query)
                add_all_cites(query_no, all_cites)
                # print(all_cites)
                c += 1
            else:
                print('present', query_no)
    print(c)

#print(c)
#df = pd.read_csv('all_cases.csv', index_col=[0])
#case_ids = df['case'].values.tolist()
#for i in case_ids:
#    res = ast.literal_eval(i)

#    print(i)
def retrieve_cites(file=all_cites):
    df = pd.read_csv(file, index_col=[0])
    case_ids = df['id'].values.tolist()
    cit = df['cites'].values.tolist()
    cites = []

    for i in cit:
        cites.append(ast.literal_eval(i))
    return case_ids, cites


def retrieve_done(id,file='map.csv'):
   try:
    id=str(id)
    df = pd.read_csv(file, index_col=[0])
    case_ids = df['id'].values.tolist()
    if int(id) in case_ids:
        return 'in_query'
    cit = df['case'].values.tolist()
    sec = df['section'].values.tolist()
    cites = []
    sections=[]
    u1=0
    for i in cit:
        u=ast.literal_eval(i)
        u1+=len(u)
        for j in u:
         cites.append(j)
        #if id in u:
        #    return 'in_case'

        #cites.append(u)
        u2=0
    for i in sec:

        u=ast.literal_eval(i)
        u2 +=len(u)
        for j in u:
         sections.append(j)
        #if id in u:
        #    return 'in_sec'
        #sections.append(ast.literal_eval(i))
    return u1, u2
   except:
       return False
print(exists(514))
def addto_all_ids(id,file=all_ids):
    data={'id':[str(id)]}
    write_csv(data,file)
def present_in_allids(id,file=all_ids):
    try:
        df = pd.read_csv(file, index_col=[0])
        case_ids = df['id'].values.tolist()
        # cases = df['case'].values.tolist()
        if int(id) in case_ids:
            return True
        return False
    except:
        return False
def candidate_create():
    file= map
    c=0
    case_ids, cites= retrieve_cites()
    for i in zip(case_ids,cites):
        sections=[]
        cases=[]
        if not present_in_allids(i[0]):
         addto_all_ids(i[0])
        if retrieve_done(i[0])=='in_query':
            continue
            print('already_retrieved')


        for j in i[1]:
            j=str(j)
            h,folder= exists(j)
            if present_in_allids(j) or retrieve_done(id=j)=='in_case':
                cases.append(j)
                print('present ...')
            elif h:
                 cases.append(j)
                 fi=os.path.join(folder, str(j)+'.json')
                 addcase(fi, str(j))
                 addto_all_ids(j)
                 print('not present...')
            elif retrieve_done(id=j)=='in_sec':
                sections.append(j)
            else:
                x,y= scraper.identify(j)
                if x=='case':
                    cases.append(j)
                    write_csv(y, all_cases)
                    addto_all_ids(j)
                if x=='act':
                    sections.append(j)
                    #addto_all_ids(j)

            c+=1

        data={'id':[str(i[0])],'section':[sections],'case':[cases]}
        write_csv(data,file)
    print(c)
#query_create()
#print(present_in_allids('12181'))
#candidate_create()
#x=retrieve_done(id=1247646)
x,y =retrieve_done(1)
print(x,y)
#print(len(list(set(y))))

#all cases, all sections-- 25166, 4839
#unique cases, unique sections-- 17555, 1588