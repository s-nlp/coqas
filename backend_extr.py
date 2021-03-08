#!/usr/bin/env python3

"""be.py: Description."""
from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flask_restful import Api, Resource, reqparse
from flask import make_response
#from nltk.tokenize import sent_tokenize, word_tokenize
import random
import json
from flask import jsonify
import json
import torch
"""Models"""


from generation.generation import diviner
from my_functions import responser
from my_functions import extractorRoberta
# Path to function with generative model
import sys
import os
module_path = os.path.dirname(__file__)
print ("module_path", module_path)
sys.path.insert(0, module_path + "/generation/gpt-2-Pytorch/")
sys.path.insert(0, module_path + "/generation/Student/")
sys.path.insert(0, module_path + "/generation/pytorch_transformers/")

print (module_path + "/generation/gpt-2-Pytorch/")
print (module_path + "/generation/Student/")

from generation.Student.cam_summarize import load_cam_model
#from text_gen_big import load_big_model
#from text_gen import load_small_model
from generation.Student.ctrl_generation import initialize_model
from generation.Student.ctrl_generation import generate_text_from_condition

model_type = "ctrl" #PUT NAME OF NEEDED MODEL
length = 100 #MODEL LENGTH
import configparser
config_parser = configparser.ConfigParser()
config_parser.read('config.ini')
config = config_parser['DEV']

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#LM_CAM = load_cam_model(device)
Cam = diviner(tp = 'cam', model = '', device = device)
print ("loaded cam")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#LM_SMALL = load_small_model(device)
#GPT2Small = diviner(tp = 'small', model = LM_SMALL, device = device)
#print ("loaded gpt2")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model, tokenizer, length = initialize_model(model_type, length, device = device)

CTRL = diviner(tp = 'ctrl', model = model, device = device, tokenizer = tokenizer)
print ("loaded ctrl")

#device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
#LM_BIG, tokenizer_big = load_big_model(device)
#GPT2Big = diviner(tp = 'big', model = LM_BIG, tokenizer = tokenizer_big, device = device)

Templ = diviner(tp = 'templates', model = '', device = device)

my_extractor = extractorRoberta(my_device = device)
print ("loaded extractors")

from snippets import get_sentence_context

def get_context_by_id_pair(id_pair):
    doc_id, sent_number = list(id_pair.items())[0]
    context = get_sentence_context(doc_id, sent_number, 1)
    context = ' '.join(context)
    return context

#my_extractor_arora = extractor_arora(my_device = 1)
#print ("loaded extractor arora")

class ReverseProxied(object):
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]

        scheme = environ.get('HTTP_X_SCHEME', '')
        if scheme:
            environ['wsgi.url_scheme'] = scheme
        return self.app(environ, start_response)


app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

reversed = True

if(reversed):
	app.wsgi_app = ReverseProxied(app.wsgi_app)
	template2 = dict(swaggerUiPrefix=LazyString(lambda : request.environ.get('HTTP_X_SCRIPT_NAME', '')))
	swagger = Swagger(app, template=template2)
else:
	swagger = Swagger(app)


api = Api(app)

@app.route('/')

#def index():

def make_response_with_exception_string(exception_string):
    response = make_response(jsonify(full_answer = exception_string, spans = [], text_spans = []))
    response.headers['content-type'] = 'application/json'
    return response
    

class Answerer_cam(Resource):
    def post(self):
        try:
            input_string = request.get_data().decode('UTF-8')
            my_extractor.from_string(input_string)
            print ("9")
            my_responser = responser()
            print ("9")
            try:
                obj1, obj2, predicates, aspects = my_extractor.get_params()
            except:
                e = sys.exc_info()[0]
                print ("Answerer CAM: exeption in extractor part ", str(sys.exc_info()))
                return ("Answerer CAM: exeption in extractor part " + str(sys.exc_info()))
            print ("9")
            print ("len(obj1), len(obj2)", len(obj1), len(obj2))
            print ("obj1, obj2, predicates", obj1, obj2, predicates)
            print ("spans", my_extractor.spans)
            if (len(obj1) > 0 and len(obj2) > 0):
                response =  my_responser.get_response(first_object = obj1, second_object = obj2, fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
                try:
                    response_json = response.json()
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer CAM: exeption in response to ltdemos.informatik.uni-hamburg.de ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer CAM: exeption in response to ltdemos.informatik.uni-hamburg.de " + str(sys.exc_info()))
                try:
                    my_diviner = Cam
                    print (1)
                    my_diviner.create_from_json(response_json, predicates)
                    print (2)
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer CAM: exeption in diviner ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer CAM: exeption in diviner " + str(sys.exc_info()))
                try:
                    answer = my_diviner.generate_advice()
                    print ("answer0", answer)
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer CAM: exeption in answer generation ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer CAM: exeption in answer generation " + str(sys.exc_info()))
            elif (len(obj1) > 0 and len(obj2) == 0):
                print ("len(obj1) > 0 and len(obj2) == 0")
                response =  my_responser.get_response(first_object = obj1, second_object = 'and', fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
                try:
                    response_json = response.json()
                    my_diviner = Cam
                    my_diviner.create_from_json(response_json, predicates)
                    answer = my_diviner.generate_advice(is_object_single = True)
                    print ("answer1", answer)  
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer CAM: exeption in response to ltdemos.informatik.uni-hamburg.de ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer CAM: exeption in response to ltdemos.informatik.uni-hamburg.de " + str(sys.exc_info()))
            else:
                answer = "We can't recognize objects for comparision"
            input_question_spans = my_extractor.spans
            my_extractor.from_string(answer)
            obj1, obj2, predicates, aspects = my_extractor.get_params()
            response = make_response(jsonify(full_answer = answer, spans = input_question_spans, text_spans = my_extractor.spans))
            response.headers['content-type'] = 'application/json'
            return response
        except:
            e = sys.exc_info()[0] 
            print ("Answerer CAM: exeption", str(sys.exc_info()))
            return make_response_with_exception_string("Answerer CAM: exeption in response to ltdemos.informatik.uni-hamburg.de " + str(sys.exc_info()))

#class extract_objects(Resource)
    
class Answerer_CTRL(Resource):
    def post(self):
        
        start_symbol = "Links "
        model_type = "ctrl" 
        length = 100 
        repetition_penalty = 1.2
        temperature = 0.2
        stop_token = None
        num_return_sequences = 5
        
        try:
            input_string = request.get_data().decode('UTF-8')
            print ("input string ", input_string)
            my_extractor.from_string(input_string)
            print ("9")
            my_responser = responser()
            print ("9")
            try:
                obj1, obj2, predicates, aspects = my_extractor.get_params()
            except:
                e = sys.exc_info()[0]
                print ("Answerer CTRL: exeption in extractor part ", str(sys.exc_info()))
                return ("Answerer CTRL: exeption in extractor part " + str(sys.exc_info()))
            
            answer = generate_text_from_condition(model, tokenizer, length, "Links " + input_string, repetition_penalty, temperature, num_return_sequences, 'ctrl', stop_token="<|endoftext|>")[0]
            input_question_spans = my_extractor.spans
            my_extractor.from_string(answer)
            obj1, obj2, predicates, aspects = my_extractor.get_params()
            response = make_response(jsonify(full_answer = answer, spans = input_question_spans, text_spans = my_extractor.spans))
            response.headers['content-type'] = 'application/json'
            return response
        except:
            e = sys.exc_info()[0] 
            print ("Answerer CTRL: exeption", str(sys.exc_info()))
            return make_response_with_exception_string("Answerer CAM: exeption in response to ltdemos.informatik.uni-hamburg.de " + str(sys.exc_info()))
        
class Answerer_snippets(Resource):
    def post(self):
        try:
            input_string  = request.get_data().decode('UTF-8')
            print ("input string ", input_string)
            my_extractor.from_string(input_string)
            print ("9")
            my_responser = responser()
            try:
                obj1, obj2, predicates, aspects = my_extractor.get_params()
            except:
                e = sys.exc_info()[0] 
                print ("Answerer Snippets: exeption in object extraction ", str(sys.exc_info()))
                return make_response_with_exception_string("Answerer TEMPL: exeption in object extraction " + str(sys.exc_info()))
            print ("len(obj1), len(obj2)", len(obj1), len(obj2))
            print ("obj1, obj2, predicates", obj1, obj2, predicates)
            if (len(obj1) > 0 and len(obj2) > 0):
                response =  my_responser.get_response(first_object = obj1, second_object = obj2, fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
                try:
                    response_json = response.json()
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer TEMPL: exeption in response to ltdemos.informatik.uni-hamburg.de ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer TEMPL: exeption in response to ltdemos.informatik.uni-hamburg.de " + str(sys.exc_info()))
                try:
                    if (response_json['object1']['name'] == response_json['winner']):
                        id_pair_win = response_json['object1']['sentences'][0]['id_pair']
                        id_pair_lose = response_json['object2']['sentences'][0]['id_pair']
                    else:
                        id_pair_win = response_json['object2']['sentences'][0]['id_pair']
                        id_pair_lose = response_json['object1']['sentences'][0]['id_pair']
                    str_win = get_context_by_id_pair(id_pair_win)
                    str_loose = get_context_by_id_pair(id_pair_lose)
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer snippets: exeption ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer snippet: exeption in genration " + str(sys.exc_info()))
                try:
                    answer = str_win + '/n' + str_loose
                    print ("answer0", answer)
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer snippets: exeption in answer generation ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer snippets: exeption in answer generation " + str(sys.exc_info()))
            elif (len(obj1) > 0 and len(obj2) == 0):
                print ("len(obj1) > 0 and len(obj2) == 0")
                response =  my_responser.get_response(first_object = obj1, second_object = 'and', fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
                try:
                    response_json = response.json()
                    my_diviner = Templ
                    my_diviner.create_from_json(response_json, predicates)
                    answer = "We can't extract 2 objects for comparison"
                    print ("answer1", answer)  
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer TEMPL: exeption in response to ltdemos.informatik.uni-hamburg.de ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer TEMPL: exeption in response to ltdemos.informatik.uni-hamburg.de " + str(sys.exc_info()))
            else:
                answer = "We can't recognize objects for comparision"
            input_question_spans = my_extractor.spans
            my_extractor.from_string(answer)
            obj1, obj2, predicates, aspects = my_extractor.get_params()
            print ("input_question_spans", input_question_spans)
            print ("output_spans", my_extractor.spans)
            response = make_response(jsonify(full_answer = answer, spans = input_question_spans, text_spans = my_extractor.spans))
            response.headers['content-type'] = 'application/json'
            return response
        except:
            e = sys.exc_info()[0] 
            print ("Answerer Template: exeption", str(sys.exc_info()))
            return make_response_with_exception_string("Answerer Templates: exeption " + str(sys.exc_info()))


class Answerer_templates(Resource):
    def post(self):
        try:
            input_string  = request.get_data().decode('UTF-8')
            print ("input string ", input_string)
            my_extractor.from_string(input_string)
            print ("9")
            my_responser = responser()
            try:
                obj1, obj2, predicates, aspects = my_extractor.get_params()
            except:
                e = sys.exc_info()[0] 
                print ("Answerer Snippets: exeption in object extraction ", str(sys.exc_info()))
                return make_response_with_exception_string("Answerer TEMPL: exeption in object extraction " + str(sys.exc_info()))
            print ("len(obj1), len(obj2)", len(obj1), len(obj2))
            print ("obj1, obj2, predicates", obj1, obj2, predicates)
            if (len(obj1) > 0 and len(obj2) > 0):
                response =  my_responser.get_response(first_object = obj1, second_object = obj2, fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
                try:
                    response_json = response.json()
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer TEMPL: exeption in response to ltdemos.informatik.uni-hamburg.de ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer TEMPL: exeption in response to ltdemos.informatik.uni-hamburg.de " + str(sys.exc_info()))
                try:
                    my_diviner = Templ
                    print (1)
                    my_diviner.create_from_json(response_json, predicates)
                    print (2)
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer TEMPL: exeption in diviner ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer TEMPL: exeption in diviner " + str(sys.exc_info()))
                try:
                    answer = my_diviner.generate_advice()
                    print ("answer0", answer)
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer TEMPL: exeption in answer generation ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer TEMPL: exeption in answer generation " + str(sys.exc_info()))
            elif (len(obj1) > 0 and len(obj2) == 0):
                print ("len(obj1) > 0 and len(obj2) == 0")
                response =  my_responser.get_response(first_object = obj1, second_object = 'and', fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
                try:
                    response_json = response.json()
                    my_diviner = Templ
                    my_diviner.create_from_json(response_json, predicates)
                    answer = my_diviner.generate_advice(is_object_single = True)
                    print ("answer1", answer)  
                except:
                    e = sys.exc_info()[0] 
                    print ("Answerer TEMPL: exeption in response to ltdemos.informatik.uni-hamburg.de ", str(sys.exc_info()))
                    return make_response_with_exception_string("Answerer TEMPL: exeption in response to ltdemos.informatik.uni-hamburg.de " + str(sys.exc_info()))
            else:
                answer = "We can't recognize objects for comparision"
            input_question_spans = my_extractor.spans
            my_extractor.from_string(answer)
            obj1, obj2, predicates, aspects = my_extractor.get_params()
            print ("input_question_spans", input_question_spans)
            print ("output_spans", my_extractor.spans)
            response = make_response(jsonify(full_answer = answer, spans = input_question_spans, text_spans = my_extractor.spans))
            response.headers['content-type'] = 'application/json'
            return response
        except:
            e = sys.exc_info()[0] 
            print ("Answerer Template: exeption", str(sys.exc_info()))
            return make_response_with_exception_string("Answerer Templates: exeption " + str(sys.exc_info()))


api.add_resource(Answerer_CTRL, '/ctrl')
api.add_resource(Answerer_snippets, '/snippets')
api.add_resource(Answerer_cam, '/cam')
#api.add_resource(Extractor1, '/extractor')


#app.jinja_env.auto_reload = True
#app.config['TEMPLATES_AUTO_RELOAD'] = True
app.run(host='0.0.0.0', port=int(config["backend_port"]))