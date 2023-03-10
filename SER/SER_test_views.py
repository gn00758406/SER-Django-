#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
from django.shortcuts import render
from sentence_transformers import SentenceTransformer, util
import gc
import csv
import os
import logging
import argparse
import random
import pickle
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import utils
from tokenizations import official_tokenization as tokenization
from models.google_albert_pytorch_modeling import AlbertConfig, AlbertForMultipleChoice
from models.pytorch_modeling import BertConfig,BertForSequenceClassification, BertForMultipleChoice, ALBertConfig, ALBertForMultipleChoice
from optimizations.pytorch_optimization import get_optimization, warmup_linear
import json

from elasticsearch import Elasticsearch
import argparse
import jieba.analyse


gpu_ids='0'                     
data_dir='C:/django/SER/data_self/'
task_name='QA'            
bert_config_file='C:/django/SER/check_points/pretrain_models/bert_config.json'
vocab_file='C:/django/SER/check_points/pretrain_models/vocab.txt'
output_dir='C:/django/SER/check_points/output_NDCG_120'
best_model='C:/django/SER/check_points/model_4.pt'                      
init_checkpoint = 'C:/django/SER/check_points/pretrain_models/pytorch_model.bin'
do_lower_case=True    
max_seq_length=512
do_train=False
do_eval=True
train_batch_size=8
eval_batch_size=16
learning_rate=2e-5   
schedule='warmup_linear'
weight_decay_rate=0.01
clip_norm=1.0  
num_train_epochs=2.0
warmup_proportion=0.05
no_cuda=False 
float16=False
seed=345  
local_rank=-1
y_scores=[]

output_amount=10


output_sentence_SBERT=[]
output_sentence_SBERT_id=[]
output_sentence_SBERT_label=[]


output_sentence_SER=[]
output_sentence_SER_id=[]
output_sentence_SER_label=[]


output_sentence_BM25=[]
output_sentence_BM25_id=[]
output_sentence_BM25_label=[]
Question_corpus = []
Answer_corpus=[]
top_k = 100
n_class = 2
reverse_order = False
sa_step = False
query=''

with open('C:/django/SER/elasticsearch/name_dict.json', 'r') as fp:
    name_dict = json.load(fp)
with open('C:/django/SER/elasticsearch/date_dict.json', 'r') as fp:
    date_dict = json.load(fp)

with open('C:/django/SER/elasticsearch/id_dict.json', 'r') as fp:
    id_dict = json.load(fp)
with open('C:/django/SER/elasticsearch/query_dict.json', 'r') as fp:
    query_dict = json.load(fp)
   
SBERT_model = SentenceTransformer('C:/Users/nlplab/.cache/torch/sentence_transformers/public.ukp.informatik.tu-darmstadt.de_reimers_sentence-transformers_v0.2_distiluse-base-multilingual-cased-v1.zip') 
    




class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
    


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()



    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines



def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))


    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)


        _truncate_seq_pair(tokens_a, tokens_b,max_seq_length - 3)
        
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        
        
            

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids))
        
    print('#features', len(features))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
os.makedirs(output_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

if local_rank == -1 or no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print(n_gpu)
else:
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)


tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


bert_config = BertConfig.from_json_file(bert_config_file)
model = BertForSequenceClassification(bert_config, num_labels=n_class)

if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            max_seq_length, bert_config.max_position_embeddings))

if init_checkpoint is not None:
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, init_checkpoint)
if float16:
    model.half()
model.to(device)

if local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                        output_device=local_rank)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(best_model))         
        
        
        
            
            
            
            
            
            
            
            

def SER_test_view(request):  #新增資料，資料必須驗證

    if os.path.exists(data_dir+'Question_corpus.npy') is not True or os.path.exists(data_dir+'Answer_corpus.npy') is not True or os.path.exists(data_dir+'label_list.npy') is not True:
        
        with open(data_dir+'NDCG_0816.json', 'r' ,  encoding='UTF-8') as f:
            data=json.load(f)
            

        id_set={}
        qa_set={}

        Question_corpus = []
        Answer_corpus = []
        label_list = []
        All_Retrieved_article_id_corpus = []
        Retrieved_article_id_list = []


        #os.path.exists('Retrieved_article_id_list.npy') is not True:

        ##將所有測試集問題與答案分開放進List
        
        for data_number in range(len(data)):
            
            label_list.append([])
            Retrieved_article_id_list.append([])
            for data_question in range(len(data[data_number])):
                Question_corpus.append(data[data_number][data_question]['Retrieved_question'])
                All_Retrieved_article_id_corpus.append(data[data_number][data_question]['Retrieved_article_id'])

                Retrieved_article_id_list[data_number].append(data[data_number][data_question]['Retrieved_article_id'])
                Answer_corpus.append(data[data_number][data_question]['Retrieved_answer'])
                label_list[data_number].append(data[data_number][data_question]['label'])

        ###SET會隨機 注意使用
        for h in range(len(Question_corpus)):
            id_set[Question_corpus[h]] =All_Retrieved_article_id_corpus[h]
        for o in range(len(Question_corpus)):
            qa_set[Question_corpus[o]] =Answer_corpus[o]
  
        Question_corpus=list(set(Question_corpus))
        Question_corpus=np.array(Question_corpus)
        Answer_corpus=[]
        All_Retrieved_article_id_corpus=[]
        for i in range(len(id_set)):
            All_Retrieved_article_id_corpus.append(id_set[Question_corpus[i]])
            Answer_corpus.append(qa_set[Question_corpus[i]])




            All_Retrieved_article_id_corpus=np.array(All_Retrieved_article_id_corpus)
            Retrieved_article_id_list=np.array(Retrieved_article_id_list)
            Answer_corpus=np.array(Answer_corpus)
            label_list=np.array(label_list)
            np.save('Retrieved_article_id_list.npy',Retrieved_article_id_list)
            np.save('All_Retrieved_article_id_corpus.npy',All_Retrieved_article_id_corpus)
            np.save('Question_corpus.npy',Question_corpus)
            np.save('Answer_corpus.npy',Answer_corpus)
            np.save('label_list.npy',label_list)
                    
            
    else:
        Question_corpus=np.load(data_dir+'Question_corpus.npy',allow_pickle=True)
        Answer_corpus=np.load(data_dir+'Answer_corpus.npy',allow_pickle=True)
        label_list=np.load(data_dir+'label_list.npy',allow_pickle=True)



           
     #取得表單輸入資料
    
    

    if os.path.exists(data_dir+'Question_corpus_embeddings.pt') is not True:
        Question_corpus_embeddings = SBERT_model.encode(Question_corpus, convert_to_tensor=True)
        torch.save(Question_corpus_embeddings, data_dir+'Question_corpus_embeddings.pt')
    else:
        Question_corpus_embeddings=torch.load(data_dir+'Question_corpus_embeddings.pt')
    
    

    
    

  
    eval_dataloader = None
#如果是以POST的方式才處理
    if request.method == "POST":  
        output_sentence_SBERT=[]
        output_sentence_SBERT_id=[]
        output_sentence_SBERT_label=[]
        output_sentence_SBERT_name=[]
        output_sentence_SBERT_date=[]
       
        y_scores=[]    
        Retrieved_answer=[]
        Retrieved_question=[]
        
        query = request.POST['input_sentence']
        
        query_list = np.load(data_dir+'query_list.npy',allow_pickle=True)
        if query in query_list:
            SBERT_y_result_NDCG = np.load(data_dir+'SBERT_y_result_NDCG.npy',allow_pickle=True)
            Retrieved_article_id_list = np.load(data_dir+'Retrieved_article_id_list.npy',allow_pickle=True)
            query_index =list(query_list).index(query)
            with open (data_dir+'SBERT_output_NDCG_120.json','r',encoding='utf8') as f:
                data2=json.load(f)
            
            
            query_embedding = SBERT_model.encode(query, convert_to_tensor=True)  
            cos_scores = util.pytorch_cos_sim(query_embedding, Question_corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            
            for i in range(100):
                y_scores.append(float(top_results.values[i]))
                output_sentence_SBERT.append(data2[query_index][i]['Retrieved_answer'])
                output_sentence_SBERT_id.append(list(Retrieved_article_id_list)[query_index][i])
                output_sentence_SBERT_label.append(list(SBERT_y_result_NDCG)[query_index][i])
                output_sentence_SBERT_name.append(name_dict[data2[query_index][i]['Retrieved_answer']])
                output_sentence_SBERT_date.append(date_dict[data2[query_index][i]['Retrieved_answer']])
                
                Retrieved_answer.append(data2[query_index][i]['Retrieved_answer'])
            zip_SBERT = zip(output_sentence_SBERT[:output_amount],output_sentence_SBERT_id[:output_amount],output_sentence_SBERT_label[:output_amount],output_sentence_SBERT_name[:output_amount],output_sentence_SBERT_date[:output_amount])
            
        else:
            query_embedding = SBERT_model.encode(query, convert_to_tensor=True)  
            cos_scores = util.pytorch_cos_sim(query_embedding, Question_corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            for i in range(100):   
                y_scores.append(float(top_results.values[i]))
                Retrieved_question.append(Question_corpus[int(top_results.indices[i])])
                Retrieved_answer.append(Answer_corpus[int(top_results.indices[i])])
            for j in range(output_amount):
                output_sentence_SBERT.append(Answer_corpus[int(top_results.indices[j])])
                output_sentence_SBERT_id.append(id_dict[Answer_corpus[int(top_results.indices[j])]])
                output_sentence_SBERT_label.append("unknown")
                output_sentence_SBERT_name.append(name_dict[Answer_corpus[int(top_results.indices[j])]])
                output_sentence_SBERT_date.append(date_dict[Answer_corpus[int(top_results.indices[j])]])
                
            zip_SBERT = zip(output_sentence_SBERT,output_sentence_SBERT_id,output_sentence_SBERT_label,output_sentence_SBERT_name,output_sentence_SBERT_date)
        if do_eval:
            
            class QAProcessor(DataProcessor):
                def __init__(self, data_dir):
                    self.data_dir = data_dir
                    self.Retrieved_answer = Retrieved_answer

                def get_dev_examples(self):
                    """See base class."""
                    return self._create_examples_2(self.Retrieved_answer, "dev")


                def _create_examples_2(self, Retrieved_answer, set_type):
                    """Creates examples for the training and dev sets."""
                    
                    
                    examples = []
                    for i in range(len(Retrieved_answer)):
                    
                                
                        
                        guid = "%s-%s" % (set_type, i)
                        text_a = tokenization.convert_to_unicode(query)
                        text_b = tokenization.convert_to_unicode(Retrieved_answer[i])
                    
                        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b))

                    return examples
            processor = QAProcessor(data_dir)
            eval_examples = processor.get_dev_examples()
            eval_features = convert_examples_to_features(eval_examples, max_seq_length, tokenizer)
                
            input_ids = []
            input_mask = []
            segment_ids = []
            

            for f in eval_features:
                
                input_ids.append(f.input_ids)
                input_mask.append(f.input_mask)
                segment_ids.append(f.segment_ids)
                

            all_input_ids = torch.tensor(input_ids, dtype=torch.long)
            all_input_mask = torch.tensor(input_mask, dtype=torch.long)
            all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
            if local_rank == -1:
                eval_sampler = SequentialSampler(eval_data)
            else:
                eval_sampler = DistributedSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
        

        if do_eval:
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            logits_all = []
            for input_ids, input_mask, segment_ids in tqdm(eval_dataloader):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, return_logits=True)

                logits = logits.detach().cpu().numpy()
                
                for i in range(len(logits)):
                    logits_all += [logits[i]]
                        

                
        
            
            with open(output_dir+ "/logits_self_dev789.txt", "w",encoding='utf8') as f:
                for i in range(len(logits_all)):
                    for j in range(len(logits_all[i])):
                        f.write(str(logits_all[i][j]))
                        if j == len(logits_all[i]) - 1:
                            f.write("\n")
                        else:
                            f.write(" ")
            
                    

        

        logits = pd.read_csv(output_dir+'/logits_self_dev789.txt',header=None,delimiter=" ")
        logits.columns = ['label_0','label_1']
        logits_list=[]
        for j in range(len(logits)):
            logits_list.append(logits['label_1'][j])
        del logits_all
        del logits
        torch.cuda.empty_cache()
        gc.collect()
       
        
        ##讀取Reranker 概率
        
        with open (data_dir+'SBERT_output_NDCG_120.json','r',encoding='utf8') as f:
            data2=json.load(f)
        

        rerank_amount = 100

        output_sentence_SER=[]
        output_sentence_SER_id=[]
        output_sentence_SER_label=[]
        output_sentence_SER_name=[]
        output_sentence_SER_date=[]
        
        ##開始重排
        def rerank():
            ##概率相乘計算總概率
            combine_list=[]
            for i in range(rerank_amount):
                combine_list.append(y_scores[i]*logits_list[i])
            sorted_combine_list = sorted(combine_list,reverse = True)
            
            ##依總概率重排前N筆
            combine_list_rank_index_list=[]
            for j in range(rerank_amount):
                combine_list_rank_index_list.append(list(combine_list).index(sorted_combine_list[j]))
            
            
            #把重排與未重排相加
            
            reranked_list=[]
            for i in range(rerank_amount):
                reranked_list.append(combine_list_rank_index_list[i])
            if rerank_amount!=100:
                for i in range(rerank_amount,100):
                    reranked_list.append(i)  
            
            y_true=[]
            for i in range(rerank_amount):
                y_true.append(int(data2[query_index][combine_list_rank_index_list[i]]['label']))
            if rerank_amount!=100:
                for i in range(rerank_amount,100):
                    y_true.append(int(data2[query_index][i]['label']))
                
                    
            return reranked_list,y_true
            
            
        def rerank2():
            ##概率相乘計算總概率
            combine_list=[]
            for i in range(rerank_amount):
                combine_list.append(y_scores[i]*logits_list[i])
            sorted_combine_list = sorted(combine_list,reverse = True)
            
            ##依總概率重排前N筆
            combine_list_rank_index_list=[]
            for j in range(rerank_amount):
                combine_list_rank_index_list.append(list(combine_list).index(sorted_combine_list[j]))
            
            
            #把重排與未重排相加
            
            reranked_list=[]
            for i in range(rerank_amount):
                reranked_list.append(combine_list_rank_index_list[i])
            if rerank_amount!=100:
                for i in range(rerank_amount,100):
                    reranked_list.append(i)  

                
            return reranked_list
        
        if query in query_list:
            reranked_list,y_true = rerank()
            for i in range(output_amount):
                output_sentence_SER.append(Retrieved_answer[reranked_list[i]])
                output_sentence_SER_id.append(id_dict[Retrieved_answer[reranked_list[i]]])
                output_sentence_SER_label.append(y_true[i])
                output_sentence_SER_name.append(name_dict[Retrieved_answer[reranked_list[i]]])
                output_sentence_SER_date.append(date_dict[Retrieved_answer[reranked_list[i]]])
                
                
                zip_SER = zip(output_sentence_SER,output_sentence_SER_id,output_sentence_SER_label,output_sentence_SER_name,output_sentence_SER_date)
        else:
            reranked_list = rerank2()
            for i in range(output_amount):
                output_sentence_SER.append(Retrieved_answer[reranked_list[i]])
                output_sentence_SER_id.append(id_dict[Retrieved_answer[reranked_list[i]]])
                output_sentence_SER_label.append("unknown")
                output_sentence_SER_name.append(name_dict[Retrieved_answer[reranked_list[i]]])
                output_sentence_SER_date.append(date_dict[Retrieved_answer[reranked_list[i]]])
                
                zip_SER = zip(output_sentence_SER,output_sentence_SER_id,output_sentence_SER_label,output_sentence_SER_name,output_sentence_SER_date)
            
            
    ####BM25        
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
           
        keyword=jieba.analyse.textrank(query, topK=8, withWeight=False, allowPOS=('ns','n','vn','v','a','ad','d','p','u','t','nr'))
        
        es = Elasticsearch()
        search = es.search(index="test_0816", body=body(keyword))
        output_sentence_BM25=[]
        output_sentence_BM25_id=[]
        output_sentence_BM25_label=[]
        output_sentence_BM25_name=[]
        output_sentence_BM25_date=[]
       
        bm25_label_list=np.load('C:/django/SER/elasticsearch/bm25_label_list.npy',allow_pickle=True)
        
       
        for i in range(output_amount):
            output_sentence_BM25.append(search['hits']['hits'][i]['_source']['Retrieved_answer'])
            output_sentence_BM25_id.append(search['hits']['hits'][i]['_source']['Retrieved_article_id'])
            if search['hits']['hits'][i]['_source']['Retrieved_answer'] in bm25_label_list[query_index]:
                output_sentence_BM25_label.append(bm25_label_list[query_index][search['hits']['hits'][i]['_source']['Retrieved_answer']]) 
            else :
                output_sentence_BM25_label.append(0)
            
            output_sentence_BM25_name.append(name_dict[search['hits']['hits'][i]['_source']['Retrieved_answer']])
            output_sentence_BM25_date.append(date_dict[search['hits']['hits'][i]['_source']['Retrieved_answer']])
        zip_BM25 = zip(output_sentence_BM25,output_sentence_BM25_id,output_sentence_BM25_label,output_sentence_BM25_name,output_sentence_BM25_date)
        #except:
            #zip_BM25 = []
        
        
    return render(request,"SER_test.html",locals())





