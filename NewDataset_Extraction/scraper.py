import random
import time
tic = time.time()
import sys, logging, os, shutil
import re
import math
import pickle as pkl
import pandas as pd
import numpy as np
import multiprocessing
from tqdm.notebook import tqdm_notebook
import requests, json
import bs4
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import sys
#import spacy

from selenium.webdriver.chrome.options import Options
options = Options()
options.add_argument("start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
def get_case(soup, case_id):
    output = {}
    output['doc_id'] = case_id
    doc_title = soup.find('div', attrs={'class': 'doc_title'})
    doc_title = doc_title.get_text()
    output['doc_title'] = doc_title
    try:
        doc_bench = soup.find('div', attrs={'class': 'doc_bench'})
        doc_bench = doc_bench.get_text()
        output['doc_bench'] = doc_bench

    except:
        print('')
    doc_text = []

    pointer = soup.find('div', attrs={'class': 'doc_title'})

    if (pointer == None):
        return {
            'doc_id': case_id
        }
    while True:
        pointer = pointer.findNextSibling()
        if (pointer == None):
            break
        if ('id' not in pointer.attrs.keys()):
            # if accessing some field like doc_title, doc_bench, doc_author
            continue

        # get paragraph text
        paragraph_text = ''

        for child in pointer.children:
            if (isinstance(child, bs4.element.NavigableString)):
                # is text not citation
                prev = child.findPreviousSibling()

                # if previous sibling is citation check for artifacts.
                if (prev != None and isinstance(prev, bs4.element.Tag) and prev.name == 'a'):
                    # if the text contains the following regex
                    temp_text = child.get_text()
                    temp_text = temp_text.split()
                    temp_text = ' '.join(temp_text)
                    temp_text = re.sub(r'IPC', '', temp_text)  # IPC
                    temp_text = re.sub(r'Cr.P.C', '', temp_text)  # Cr.P.C
                    temp_text = re.sub(r'\(\d+\) \d+ SCC \d+', '', temp_text)  # SCC, example case 69949024
                    temp_text = re.sub(r'\(\d+ of \d+\)', '', temp_text)  # (45 of 1860), example case 69949024
                    temp_text = re.sub(r'\[\d+\]', '', temp_text)  # [26], example case 69949024

                    paragraph_text += (temp_text + ' ')

                else:
                    # normal text
                    temp_text = child.get_text()
                    temp_text = temp_text.split()
                    temp_text = ' '.join(temp_text)
                    paragraph_text += (temp_text + ' ')

            # else citation
            elif (isinstance(child, bs4.element.Tag) and child.name == 'a'):
                # insert citation
                citation_id = int(re.findall(r'\d+', child.attrs['href'])[-1])

                # mask citation if needed
                temp_text = child.get_text()

                #if (is_section(citation_id, driver)):
                #    temp_text = temp_text.split()
                #    temp_text = ' '.join(temp_text)
                #    index = len(paragraph_text)  # this is the index at which the citation occurs
                #    paragraph_text += (temp_text + ' ')


                #else:
                index = len(paragraph_text)  # this is the index at which the citation occurs
                paragraph_text += '<CITATION_{}> '.format(citation_id)


        if (len(paragraph_text) == 0):
            continue
        if pointer.attrs['id'][0:3]=='pre':
            continue
        doc_text.append({
            'id': pointer.attrs['id'],  # paragraph id on indian kanoon website.
            'paragraph_text': paragraph_text,
        })

    output['doc_text'] = doc_text
    return output

def identify(case_id):
    output={}
    url = f'https://indiankanoon.org/doc/{case_id}/'
    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        t = soup.find('div', attrs={'class': 'acts'})
        if t:
            print("act")
            return 'act', ''
        else:

            t = soup.find('div', attrs={'class': 'judgments'})
            if t:
              print('case')
              output=get_case(soup, case_id)
              print(output)
              cas = []
              for x in output['doc_text']:
                  x["paragraph_text"] = x["paragraph_text"].replace("\n", "")
                  cas.append(x["paragraph_text"])
              d = {'id': [str(case_id)], 'case': [cas]}
              return 'case', d



        return 'failed',''
        #print(t)
        #doc_title = doc_title.get_text()
        #output['doc_title'] = doc_title
    except:
        print(F"\nUnable to access the URL\n{url}\nSome error occured\n")
        return 'failed',''
#case_id='128647346'
#identify(case_id)