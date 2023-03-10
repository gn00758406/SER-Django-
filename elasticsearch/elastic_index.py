#####檔案放入#######
from elasticsearch import Elasticsearch, helpers
import json



def read_data():
    spots = []
    
    with open("BM25_提供檢索.json", 'r' ,  encoding='UTF-8') as f:
        x=json.load(f)
        
    for i in range(len(x)):

        yield x[i]
        


INDEX_NAME = 'test_0816'
DOC_TYPE = '_doc'
es = Elasticsearch()
success, _ = helpers.bulk(
        es, read_data(), index=INDEX_NAME, doc_type=DOC_TYPE, ignore=400)
print(INDEX_NAME)
print('success:', success)
