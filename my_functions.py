def do_sum(number1, number2):
    return number1 + number2

import os.path
import torch
import sys
import spacy
import re

# for extractor class
sys.path.append('/home/vika/targer')
sys.path.append('/notebook/cqas')
sys.path.append('/notebook/bert_sequence_sayankotor/src')

# for responser class
import json
import requests

# for generate answer
sys.path.insert(0, "/notebook/cqas/generation")
#from generation.generation import diviner

import os
current_directory_path = os.path.dirname(os.path.realpath(__file__))

# pathes to pretrained extraction model

PATH_TO_PRETRAINED = '/external_pretrained_models/'
MODEL_NAMES = ['bertttt.hdf5']

# Roberta

from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder

from allennlp.models import SimpleTagger
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.predictors import SentenceTaggerPredictor


from allennlp.data.dataset_readers import Conll2003DatasetReader
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader

def load(checkpoint_fn, gpu=-1):
    if not os.path.isfile(PATH_TO_PRETRAINED + checkpoint_fn):
        raise ValueError('Can''t find tagger in file "%s". Please, run the main script with non-empty \
                         "--save-best-path" param to create it.' % checkpoint_fn)
    tagger = torch.load(PATH_TO_PRETRAINED + checkpoint_fn)
    tagger.gpu = gpu

    tagger.word_seq_indexer.gpu = gpu # hotfix
    tagger.tag_seq_indexer.gpu = gpu # hotfix
    if hasattr(tagger, 'char_embeddings_layer'):# very hot hotfix
        tagger.char_embeddings_layer.char_seq_indexer.gpu = gpu # hotfix
    tagger.self_ensure_gpu()
    return tagger

def create_sequence_from_sentence(str_sentences):
    return [re.findall(r"[\w']+|[.,!?;]", str_sentence.lower()) for str_sentence in str_sentences]

class extractorRoberta:
    def __init__(self, my_device = torch.device('cuda:2'), model_name = 'roberta.hdf5', model_path = current_directory_path + '/external_pretrained_models/'):
        self.answ = "UNKNOWN ERROR"
        self.model_name = model_name
        self.model_path = model_path
        self.first_object = ''
        self.second_object = ''
        self.predicates = ''
        self.aspects = ''
        cuda_device = my_device
        self.spans = [] # we can't use set because span object is dict and dict is unchashable. We add function add_span to keep non-repeatability
        try:
            print (self.model_path + self.model_name)
            print (model_path + "vocab_dir")
            vocab= Vocabulary.from_files(model_path + "vocab_dir")
            BERT_MODEL = 'google/electra-base-discriminator'
            embedder = PretrainedTransformerMismatchedEmbedder(model_name=BERT_MODEL)
            text_field_embedder = BasicTextFieldEmbedder({'tokens': embedder})
            seq2seq_encoder = PassThroughEncoder(input_dim=embedder.get_output_dim())
            print ("encoder loaded")
            self.indexer = PretrainedTransformerMismatchedIndexer(model_name=BERT_MODEL)
            print ("indexer loaded")
            self.model = SimpleTagger(text_field_embedder=text_field_embedder, 
                      vocab=vocab, 
                      encoder=seq2seq_encoder,
                      calculate_span_f1=True,
                      label_encoding='IOB1').cuda(device=cuda_device)
            self.model.load_state_dict(torch.load(self.model_path + self.model_name))
            print ("model loaded")
            self.reader = Conll2003DatasetReader(token_indexers={'tokens': self.indexer})
            print ("reader loaded")
        except:
            e = sys.exc_info()[0]
            print ("exeption while mapping to gpu in extractor ", e)
            raise RuntimeError("Init extractor: can't map to gpu. Maybe it is OOM")
        try:
            self.predictor = SentenceTaggerPredictor(self.model, self.reader)
        except:
            e = sys.exc_info()[0]
            print ("exeption in creating predictor ", e)
            raise RuntimeError("Init extractor: can't map to gpu. Maybe it is WTF")
            
    def add_span(self, span_obj):
        if span_obj not in self.spans:
            self.spans.append(span_obj)
      
    def get_words():
        return self.words
    
    def get_tags():
        return self.tags
        
    def from_string(self, input_sentence):
        self.input_str = input_sentence
        self.first_object = ''
        self.second_object = ''
        self.predicates = ''
        self.aspects = ''
        self.spans = []
        
    def get_objects_predicates(self, list_words, list_tags):
        starts = set()
        obj_list = []
        pred_list = []
        asp_list = []
        for ind, elem in enumerate(list_tags):
            if elem == 'B-Object':
                obj_list.append(list_words[ind])
                start = self.input_str.find(list_words[ind])
                while (start in starts):
                    print (1)
                    print ("ind, word", ind, list_words[ind])
                    print ("old start, starts", start, starts)
                    print ("string ", self.input_str[start + len(list_words[ind]):])
                    start = self.input_str[start + len(list_words[ind]):].find(list_words[ind]) + start + len(list_words[ind])
                    print ("new start", start)
                if (start != -1 and {'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" } not in self.spans):
                    self.spans.append({'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" })
                    starts.add(start)
            if elem == 'I-Object':
                start = self.input_str.find(list_words[ind])
                while (start in starts):
                    start = self.input_str[start:].find(list_words[ind]) + start + len(list_words[ind])
                if (start != -1 and {'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" } not in self.spans):
                    self.spans.append({'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" })
                    starts.add(start)
            if elem == 'B-Predicate':
                print (2)
                print ("ind, word", ind, list_words[ind])
                print ("old start, starts", start, starts)
                print ("string ", self.input_str[start + len(list_words[ind]):])
                pred_list.append(list_words[ind])
                start = self.input_str.find(list_words[ind] + " ")
                while (start in starts):
                    start = self.input_str[start  + len(list_words[ind]):].find(list_words[ind]) + start + len(list_words[ind])
                if (start != -1 and { 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" } not in self.spans):
                    self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" })
                    starts.add(start)
            if elem == 'I-Predicate':
                start = self.input_str.find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" })
                while (start in starts):
                    start = self.input_str[start:].find(list_words[ind]) + start + len(list_words[ind])
                if (start != -1 and { 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" } not in self.spans):
                    self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" })
                    starts.add(start)
            if elem == 'I-Aspect':
                asp_list.append(list_words[ind])
                start = self.input_str.find(list_words[ind])
                while (start in starts):
                    start = self.input_str[start:].find(list_words[ind]) + start + len(list_words[ind])
                if (start != -1 and {'end': start + len(list_words[ind]), 'start': start, 'type': "ASP" } not in self.spans):
                    self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "ASP" })
                    starts.add(start)
            if elem == 'B-Aspect':
                asp_list.append(list_words[ind])
                start = self.input_str.find(list_words[ind])
                while (start in starts):
                    start = self.input_str[start:].find(list_words[ind]) + start + len(list_words[ind])
                if (start != -1 and {'end': start + len(list_words[ind]), 'start': start, 'type': "ASP" } not in self.spans):
                    self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "ASP" })
                    starts.add(start)
        return obj_list, pred_list, asp_list
    
    def extract_objects_predicates(self, input_string):
        preds = self.predictor.predict(input_string)
        print ("extract_objects_predicates tags", preds['tags'])
        print ("extract_objects_predicates words", preds['words'])
        objects, predicates, aspects = self.get_objects_predicates(preds['words'], preds['tags'])
        print (objects)
        print (predicates)
        print (aspects)
        self.predicates = predicates
        self.aspects = aspects
        print ("len(objects)", len(objects))
        if len(objects) >= 2:
            self.first_object = objects[0]
            self.second_object = objects[1]
        else: # try to use spacy
            self.answ = "We can't recognize two objects for compare 2" 
                
    def get_params(self):
        print ("in extractor get params 0")
        print ("self prredicates ", self.predicates)
        self.extract_objects_predicates(self.input_str)
        #except:
            #raise RuntimeError("Can't map to gpu. Maybe it is OOM")
        return self.first_object.strip(".,!/?"), self.second_object.strip(".,!/?"), self.predicates, self.aspects
    
    
class responser:
    def __init__(self):
        self.URL = 'http://ltdemos.informatik.uni-hamburg.de/cam-api'
        self.proxies = {"http": "http://185.46.212.97:10015/","https": "https://185.46.212.98:10015/",}
        
    def get_response(self, first_object, second_object, fast_search=True, 
               aspects=None, weights=None):
        print ("aspects", aspects)
        print ("weights", weights)
        num_aspects = len(aspects) if aspects is not None else 0
        num_weights = len(weights) if weights is not None else 0
        if num_aspects != num_weights:
            raise ValueError(
                "Number of weights should be equal to the number of aspects")
        params = {
            'objectA': first_object,
            'objectB': second_object,
            'fs': str(fast_search).lower()
        }
        if num_aspects:
            params.update({'aspect{}'.format(i + 1): aspect 
                           for i, aspect in enumerate(aspects)})
            params.update({'weight{}'.format(i + 1): weight 
                           for i, weight in enumerate(weights)})
        print ("get url")
        print ("params", params)
        response = requests.get(url=self.URL, params=params, timeout=70)
        return response
    
def answerer(input_string, tp = 'big'):
    my_extractor = extractorRoberta()
    my_extractor.from_string(input_string)
    my_responser = responser()
    obj1, obj2, predicates, aspects = my_extractor.get_params()
    print ("len(obj1), len(obj2)", len(obj1), len(obj2))
    print ("obj1, obj2, predicates", obj1, obj2, predicates)
    if (len(obj1) > 0 and len(obj2) > 0):
        response =  my_responser.get_response(first_object = obj1, second_object = obj2, fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
        try:
            response_json = response.json()
        except:
            return ("smth wrong in response, please try again")
        try:
            my_diviner = diviner(tp = tp)
            print (1)
            my_diviner.create_from_json(response_json, predicates)
            print (2)
        except:
            return ("smth wrong in diviner, please try again")
        try:
            answer = my_diviner.generate_advice()
            print ("answer", answer)
            #del my_extractor,my_diviner, my_responser
            return answer
        except:
            #del my_extractor,my_diviner, my_responser
            return ("smth wrong in answer generation, please try again")
    elif (len(obj1) > 0 and len(obj2) == 0):
        print ("len(obj1) > 0 and len(obj2) == 0")
        response =  my_responser.get_response(first_object = obj1, second_object = 'and', fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
        try:
            response_json = response.json()
            my_diviner = diviner(tp = tp)
            my_diviner.create_from_json(response_json, predicates)
            answer = my_diviner.generate_advice(is_object_single = True)
            print ("answer", answer)
            #del my_extractor,my_diviner, my_responser
            return answer  
        except:
            #del my_extractor,my_diviner, my_responser
            return ("smth wrong in response, please try again")
    else:
        return ("We can't recognize objects for comparision")
    
    
