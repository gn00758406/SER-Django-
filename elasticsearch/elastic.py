from elasticsearch import Elasticsearch
import argparse
import jieba.analyse

def body(keyword):
  if len(keyword)==0:
    body={"size": 100,"query": {
      "bool": {
        "should": [
          { "multi_match": {"query":query,"fields":["Retrieved_question"]}}
          ]}}}

  elif len(keyword)==1:
    body={"size": 100,"query": {
      "bool": {
        "should": [
          { "multi_match": {"query":query,"fields":["Retrieved_question"]}},
          {"multi_match": {"query":keyword[0],"fields": ["Retrieved_question"],"boost": 3}}]}}}
  elif len(keyword)==2:
    body={"size": 100,"query": {
      "bool": {
        "should": [
          { "multi_match": {"query":query,"fields":["Retrieved_question"]}},
          {"multi_match": {"query":keyword[0],"fields": ["Retrieved_question"],"boost": 3}},
          {"multi_match": {"query":keyword[1],"fields":["Retrieved_question"], "boost": 3}}]}}}


  elif len(keyword)==3:
    body={"size": 100,"query": {"bool": {
        "should": [
          { "multi_match": {"query":query,"fields":["Retrieved_question"]}},
            {"multi_match": {"query":keyword[0],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[1],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[2],"fields":["Retrieved_question"], "boost": 3}}]}}}

  elif len(keyword)==4:
    body={"size": 100,"query": {"bool": {
        "should": [
          { "multi_match": {
            "query":query,
            "fields":["Retrieved_question"]}},
            {"multi_match": {"query":keyword[0],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[1],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[2],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[3],"fields":["Retrieved_question"], "boost": 3}}]}}}
  elif len(keyword)==5:
    body={"size": 100,"query": {"bool": {
        "should": [
          { "multi_match": {
            "query":query,
            "fields":["Retrieved_question"]}},
            {"multi_match": {"query":keyword[0],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[1],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[2],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[3],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[4],"fields":["Retrieved_question"], "boost": 3}}]}}}

  elif len(keyword)==6:
    body={"size": 100,"query": {"bool": {
        "should": [
          { "multi_match": {
            "query":query,
            "fields":["Retrieved_question"]}},
            {"multi_match": {"query":keyword[0],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[1],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[2],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[3],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[4],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[5],"fields":["Retrieved_question"], "boost": 3}}]}}}

  elif len(keyword)==7:
    body={"size": 100,"query": {"bool": {
        "should": [
          { "multi_match": {
            "query":query,
            "fields":["Retrieved_question"]}},
            {"multi_match": {"query":keyword[0],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[1],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[2],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[3],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[4],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[5],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[6],"fields":["Retrieved_question"], "boost": 3}}]}}}

  elif len(keyword)==8:
    body={"size": 100,"query": {"bool": {
        "should": [
          { "multi_match": {"query":query,"fields":["Retrieved_question"]}},
            {"multi_match": {"query":keyword[0],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[1],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[2],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[3],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[4],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[5],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[6],"fields":["Retrieved_question"], "boost": 3}},
            {"multi_match": {"query":keyword[7],"fields":["Retrieved_question"], "boost": 3}}]}}}
  return body

query="頭痛該怎麼辦"        
keyword=jieba.analyse.textrank(query, topK=8, withWeight=False, allowPOS=('ns','n','vn','v','a','ad','d','p','u','t','nr'))
output_amount=10        
es = Elasticsearch()
search = es.search(index="test", body=body(keyword))
output_sentence_BM25=[]
output_sentence_BM25_id=[]
output_sentence_BM25_label=[]
output_sentence_BM25_name=[]
output_sentence_BM25_date=[]
print(search['hits']['hits'][0]['_source'])
