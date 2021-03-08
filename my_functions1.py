def do_sum(number1, number2):
    return number1 + number2

import os.path
import torch
import sys
import spacy

import en_core_web_sm

# for extractor class
sys.path.append('/home/vika/targer')
sys.path.append('/notebook/cqas')
from src.factories.factory_tagger import TaggerFactory
from src.layers import layer_context_word_embeddings_bert

# for responser class
import json
import requests

# for generate answer
from generation.generation import diviner

import os
current_directory_path = os.path.dirname(os.path.realpath(__file__))

# pathes to pretrained extraction model

PATH_TO_PRETRAINED = '/external_pretrained_models/'
MODEL_NAMES = ['bertttt.hdf5']

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

import nltk

def create_sequence_from_sentence(str_sentences):
    return [nltk.word_tokenize(str_sentence) for str_sentence in str_sentences]


class extractor:
    def __init__(self, my_device = 6, model_name = 'bertttt.hdf5', model_path = current_directory_path + '/external_pretrained_models/'):
        self.answ = "UNKNOWN ERROR"
        self.model_name = model_name
        self.model_path = model_path
        self.first_object = ''
        self.second_object = ''
        self.predicates = ''
        self.aspects = []
        self.spans = [] # we can't use set because span object is dict and dict is unchashable. We add function add_span to keep non-repeatability
        try:
            print (my_device)
            self.model = TaggerFactory.load(self.model_path + self.model_name, my_device)
            self.model.cuda(device=my_device)
            self.model.gpu = my_device
        except:
            raise RuntimeError("Init extractor: can't map to gpu. Maybe it is OOM")
            
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
        self.aspects = []
        self.spans = []
        
    def get_objects_predicates(self, list_words, list_tags):
        obj_list = []
        pred_list = []
        for ind, elem in enumerate(list_tags):
            if elem == 'B-OBJ':
                obj_list.append(list_words[ind])
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" })
            if elem == 'I-OBJ':
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" })
            if elem == 'B-PREDFULL':
                pred_list.append(list_words[ind])
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" })
            if elem == 'I-PREDFULL':
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" })
        return obj_list, pred_list
    
    def get_aspects(self, list_words, list_tags):
        for ind, elem in enumerate(list_tags):
            if elem == 'B-ASP':
                self.aspects.append(list_words[ind])
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({'end': start + len(list_words[ind]), 'start': start, 'type': "ASP" })
            if elem == 'I-ASP':
                self.aspects.append(list_words[ind])
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "ASP" })
        return self.aspects
    
    def extract_objects_predicates(self, input_sentence):
        words = create_sequence_from_sentence([input_sentence])  
        tags = self.model.predict_tags_from_words(words)
        print ("extract_objects_predicates tags", tags[0])
        print ("extract_objects_predicates words", words[0])
        objects, predicates = self.get_objects_predicates(words[0], tags[0])
        aspects = self.get_aspects(words[0], tags[0])
        print (objects)
        print (predicates)
        print (aspects)
        self.predicates = predicates
        print ("len(objects)", len(objects))
        if len(objects) >= 2:
            self.first_object = objects[0]
            self.second_object = objects[1]
        else: # try to use spacy
            
            if len(objects) == 1:
                self.first_object = objects[0]
                self.second_object = ''
            
            print("We try to use spacy")
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(input_sentence)
            tokens = [token.text for token in doc]
            split_sent = words[0]
            print("We try to use spacy")
            if (len(self.predicates) == 0):
                for ind, token in enumerate(doc):
                    if (doc[ind].tag_ == 'JJR' or doc[ind].tag_ == 'RBR'):
                        self.predicates = doc[ind].text
                        self.add_span({'end': self.input_str.lower().find(doc[ind].text) + len(doc[ind].text), 'start': self.input_str.lower().find(doc[ind].text), 'type': "PRED" })
                        break
            print ("split_sent", split_sent)
            print ('or' in split_sent)
            if 'or' in split_sent:
                comp_elem = 'or'
            elif 'and' in split_sent:
                comp_elem = 'and'
            elif 'vs' in split_sent:
                comp_elem = 'vs'
            elif 'vs.' in split_sent:
                comp_elem = 'vs.'
            else:
                self.answ = "We can't recognize two objects for compare 0"
                return
            print ("comp_elem", comp_elem)
            print ("tokens", tokens)
            if (comp_elem in tokens):
                print ("comp elem in tokens")
                or_index = tokens.index(comp_elem)               
                if (len (doc.ents) >= 2):
                    print ("or doc ents", or_index)
                    for ent in doc.ents:
                        print ("doc ent text", ent.text, ent.start, ent.end, or_index)
                        if (ent.end == or_index):
                            self.first_object = ent.text
                            self.add_span({'end': ent.end,'start': ent.start, 'type': "OBJ" })
                        if (ent.start == or_index + 1):
                            self.second_object = ent.text
                            self.add_span({'end': ent.end, 'start': ent.start, 'type': "OBJ" })
                            

                else:
                    print ("or simple split_sent", or_index)
                    try:
                        obj1 = tokens[or_index - 1] # tokens are uppercase. self.input_str is uppercase
                        obj2 = tokens[or_index + 1]
                        print (obj1, obj2)
                        self.first_object = obj1
                        self.second_object = obj2
                        self.add_span({'end': self.input_str.find(obj1) + len(obj1), 'start': self.input_str.find(obj1), 'type': "OBJ" })
                        self.add_span({'end': self.input_str.find(obj2) + len(obj2), 'start': self.input_str.find(obj2), 'type': "OBJ" })
                    except:
                        self.answ = "We can't recognize two objects for compare 1" 
            else:
                self.answ = "We can't recognize two objects for compare 2" 
                
    def get_params(self):
        print ("in extractor get params 0")
        #try:
        self.extract_objects_predicates(self.input_str)
        #except:
            #raise RuntimeError("Can't map to gpu. Maybe it is OOM")
        return self.first_object.strip(".,!/?"), self.second_object.strip(".,!/?"), self.predicates
    
    def get_aspect(self):
        return self.aspects
        
    
class extractorAurora(extractor):
    def __init__(self, my_device = 6, model_name = 'Aurora.hdf5', model_path = current_directory_path + '/external_pretrained_models/'):
        self.answ = "UNKNOWN ERROR"
        self.model_name = model_name
        self.model_path = model_path
        self.first_object = ''
        self.second_object = ''
        self.predicates = ''
        self.spans = [] # we can't use set because span object is dict and dict is unchashable. We add function add_span to keep non-repeatability
        try:
            self.model = TaggerFactory.load(self.model_path + self.model_name, my_device)
            self.model.cuda(device=my_device)
            self.model.gpu = my_device
            print ("extract_objects_predicates gpu", self.model.gpu)
        except:
            raise RuntimeError("Init extractor: can't map to gpu. Maybe it is OOM")
            
        
    def get_objects_predicates(self, list_words, list_tags):
        obj_list = []
        pred_list = []
        asp_list = []
        for ind, elem in enumerate(list_tags):
            if elem == 'PROD1':
                obj_list.append(list_words[ind])
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" })
            if elem == 'PROD2':
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" })
            if elem == 'PRED':
                pred_list.append(list_words[ind])
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" })
            if elem == 'ASP':
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "ASP" })
        return obj_list, pred_list
            
    
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
        response = requests.get(url=self.URL, params=params)
        return response
    
def answerer(input_string, tp = 'big'):
    my_extractor = extractor()
    my_extractor.from_string(input_string)
    my_responser = responser()
    obj1, obj2, predicates = my_extractor.get_params()
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
            #del my_extractor,my_diviner, my_responser
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
            my_diviner = diviner(tp = "big")
            my_diviner.create_from_json(response_json, predicates)
            answer = my_diviner.generate_advice(is_object_single = True)
            print ("answer", answer)
            #del my_extractor,my_diviner, my_responser
            return answer  
        except:
            #del my_extractor,my_diviner, my_responser
            return ("smth wrong in response, please try again")
    elif (len(obj2) > 0 and len(obj1) == 0):
        print ("len(obj2) > 0 and len(obj1) == 0")
        response =  my_responser.get_response(first_object = obj2, second_object = 'and', fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
        try:
            response_json = response.json()
            my_diviner = diviner(tp = "big")
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
    
    
class extractorArora(extractor):
    def __init__(self, my_device = 6, model_name = 'aurora_berts_simple.hdf5', model_path = current_directory_path + '/external_pretrained_models/'):
        self.answ = "UNKNOWN ERROR"
        self.model_name = model_name
        self.model_path = model_path
        self.first_object = ''
        self.second_object = ''
        self.predicates = ''
        self.spans = [] # we can't use set because span object is dict and dict is unchashable. We add function add_span to keep non-repeatability
        try:
            self.model = TaggerFactory.load(self.model_path + self.model_name, my_device)
            self.model.cuda(device=my_device)
            self.model.gpu = my_device
            print ("extract_objects_predicates gpu", self.model.gpu)
        except:
            e = sys.exc_info()[0]
            print ("exeption while mapping to gpu in extractorArora ", e)
            raise RuntimeError("Init extractor: can't map to gpu. Maybe it is OOM")
            
        
    def get_objects_predicates(self, list_words, list_tags):
        obj_list = []
        pred_list = []
        asp_list = []
        for ind, elem in enumerate(list_tags):
            if elem == 'PROD1':
                obj_list.append(list_words[ind])
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" })
            if elem == 'PROD2':
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "OBJ" })
            if elem == 'PRED':
                pred_list.append(list_words[ind])
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "PRED" })
            if elem == 'ASP':
                start = self.input_str.lower().find(list_words[ind])
                self.spans.append({ 'end': start + len(list_words[ind]), 'start': start, 'type': "ASP" })
        return obj_list, pred_list
    
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
    my_extractor = extractor()
    my_extractor.from_string(input_string)
    my_responser = responser()
    obj1, obj2, predicates = my_extractor.get_params()
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
            my_diviner = diviner(tp = "big")
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
    
    
class extractorArtemArora(extractorArora):
     def __init__(self, my_device = 1, model_name = "artem_bert_arora.hdf5", model_path = current_directory_path + '/external_pretrained_models/'):
        self.answ = "UNKNOWN ERROR"
        self.model_name = model_name
        self.model_path = model_path
        self.first_object = ''
        self.second_object = ''
        self.predicates = ''
        self.spans = [] # we can't use set because span object is dict and dict is unchashable. We add function add_span to keep non-repeatability
        try:
            print (self.model_path + self.model_name)
            tagger = torch.load(self.model_path + self.model_name)
            self.model = tagger
        except: # catch *all* exceptions
            e = sys.exc_info()[0]
            print ("exeption while extracting to gpu ", str(sys.exc_info()))
        try:
            print (111)
            self.model.cuda(device=my_device)
            self.model.gpu = my_device
            print (111)
            print ("extract_objects_predicates gpu", str(sys.exc_info()))
        except:
            e = sys.exc_info()[0]
            print (type(sys.exc_info()))
            print (type(e))
            print (str(sys.exc_info()))
            print (str(e))
            print ("exeption while mapping to gpu in SeqBert ", str(sys.exc_info()))
            raise RuntimeError("Init extractor. Maybe it is OOM")

     def predict_string(self, tokens):
        print ("tokens")
        print (tokens)
        _, max_len, token_ids, token_masks, bpe_masks = self.model._make_tokens_tensors([tokens], self.model._max_len)
        label_ids = None
        loss_masks = None

        with torch.no_grad():
            token_ids = token_ids.cuda(device=self.model.gpu)
            token_masks = token_masks.cuda(device=self.model.gpu)
            #loss_masks = loss_masks.cuda(device=self.model.gpu)
            
            print ("x")

            logits = self.model._bert_model(token_ids, 
                                      token_type_ids=None,
                                      attention_mask=token_masks,
                                      labels=label_ids,
                                      loss_mask=loss_masks)
            print ("xxx")
            logits = logits[0]
            print ("xxxx")
            b_preds, prob = self.model._logits_to_preds(logits.cpu(), bpe_masks, tokens)

        print ("bpreds", b_preds)
        return b_preds
    
            
     def extract_objects_predicates(self, input_sentence):
        words = create_sequence_from_sentence([input_sentence])   
        tags = self.predict_string(words[0])
        print ("extract_objects_predicates tags", tags[0])
        print ("extract_objects_predicates words", words[0])
        objects, predicates = self.get_objects_predicates(words[0], tags[0])
        print (objects)
        print (predicates)
        self.predicates = predicates
        print ("len(objects)", len(objects))
        if len(objects) >= 2:
            self.first_object = objects[0]
            self.second_object = objects[1]
        else: # try to use spacy
            if len(objects) == 1:
                self.first_object = objects[0]
                self.second_object = ''
            print("We try to use spacy")
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(input_sentence)
            tokens = [token.text for token in doc]
            split_sent = words[0]
            
            if (len(self.predicates) == 0):
                print ("pand")
                for ind, token in enumerate(doc):
                    if (doc[ind].tag_ == 'JJR' or doc[ind].tag_ == 'RBR'):
                        print ("pand 0")
                        self.predicates = [doc[ind].text]
                        self.add_span({'end': self.input_str.lower().find(doc[ind].text) + len(doc[ind].text), 'start': self.input_str.lower().find(doc[ind].text), 'type': "PRED" })
                        break
            
            if 'or' in split_sent:
                comp_elem = 'or'
            elif 'vs' in split_sent:
                comp_elem = 'vs'
            elif 'vs.' in split_sent:
                comp_elem = 'vs.'
            else:
                self.answ = "We can't recognize two objects for compare 0"
                return
            print ("comp_elem", comp_elem)
            print ("tokens", tokens)
            if (comp_elem in tokens):
                print ("comp elem in tokens")
                or_index = tokens.index(comp_elem)               
                if (len (doc.ents) >= 2):
                    for ent in doc.ents:
                        if (ent.end == or_index):
                            self.first_object = ent.text
                            self.add_span({'end': ent.end,'start': ent.start, 'type': "OBJ" })
                        if (ent.start == or_index + 1):
                            self.second_object = ent.text
                            self.add_span({'end': ent.end, 'start': ent.start, 'type': "OBJ" })
                            

                else:
                    print ("or simple split_sent", or_index)
                    try:
                        obj1 = tokens[or_index - 1] # tokens are uppercase. self.input_str is uppercase
                        obj2 = tokens[or_index + 1]
                        print (obj1, obj2)
                        self.first_object = obj1
                        self.second_object = obj2
                        self.add_span({'end': self.input_str.find(obj1) + len(obj1), 'start': self.input_str.find(obj1), 'type': "OBJ" })
                        self.add_span({'end': self.input_str.find(obj2) + len(obj2), 'start': self.input_str.find(obj2), 'type': "OBJ" })
                    except:
                        self.answ = "We can't recognize two objects for compare 1" 
            else:
                self.answ = "We can't recognize two objects for compare 2" 