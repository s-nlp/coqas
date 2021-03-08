import sys
import torch
sys.path.insert(0, "/notebook/cqas")
sys.path.insert(0, "/notebook/cqas/generation/gpt-2-Pytorch")
sys.path.insert(0, "/notebook/cqas/generation/Student")
sys.path.insert(0, "/notebook/cqas/generation/pytorch_transformers")

from generation.generation import diviner
from my_functions import extractor
from my_functions import responser

def generate_one_answer_with_defined_objects(obj1, obj2, my_diviner):
    # this is an extention of generate_one_answer function
    # for ablation study we should generate and further evaluate 
    # reaponses based on obj1, obj2 from table
    my_responser = responser()
    response =  my_responser.get_response(first_object = obj1, second_object = obj2, fast_search=True, aspects = [], weights = [])
    print ("response", response)
    try:
        response_json = response.json()
        print ("11")
        my_diviner.create_from_json(response_json, [])
        print ("22")
        answer = my_diviner.generate_advice(is_object_single = False)
        print ("33")
        print ("answer1", answer)  
    except:
        answer = "smth wrong in response, please try again"
    return answer


def generate_one_answer(input_string, my_extractor, my_diviner):
    my_extractor.from_string(input_string)
    my_responser = responser()
    try:
        obj1, obj2, predicates = my_extractor.get_params()
    except:
        return ("smth wrong in extractor, please try again")
    #print ("len(obj1), len(obj2)", len(obj1), len(obj2))
    print ("obj1, obj2, predicates", obj1, obj2, predicates)
    #print ("spans", my_extractor.spans)
    if (len(obj1) > 0 and len(obj2) > 0):
        response =  my_responser.get_response(first_object = obj1, second_object = obj2, fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
        try:
            response_json = response.json()
        except:
            return ("smth wrong in response, please try again")
        try:
            #my_diviner = Cam
            print (1)
            my_diviner.create_from_json(response_json, predicates)
            print (2)
        except:
            return ("smth wrong in diviner, please try again")
        try:
            answer = my_diviner.generate_advice()
            #print ("answer0", answer)
        except:
            return ("smth wrong in answer generation, please try again")
    elif (len(obj1) > 0 and len(obj2) == 0):
        print ("len(obj1) > 0 and len(obj2) == 0")
        response =  my_responser.get_response(first_object = obj1, second_object = 'and', fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
        try:
            response_json = response.json()
            #my_diviner = Cam
            my_diviner.create_from_json(response_json, predicates)
            answer = my_diviner.generate_advice(is_object_single = True)
            #print ("answer1", answer)  
        except:
            answer = "smth wrong in response, please try again"
    elif (len(obj2) > 0 and len(obj1) == 0):
        print ("len(obj2) > 0 and len(obj1) == 0")
        response =  my_responser.get_response(first_object = obj2, second_object = 'and', fast_search=True, aspects = predicates, weights = [1 for predicate in predicates])
        try:
            response_json = response.json()
            #my_diviner = Cam
            my_diviner.create_from_json(response_json, predicates)
            answer = my_diviner.generate_advice(is_object_single = True)
            #print ("answer1", answer)  
        except:
            answer = "smth wrong in response, please try again"
    else:
        answer = "We can't recognize objects for comparision"
    return answer