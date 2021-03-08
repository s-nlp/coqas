import os.path
import torch
import argparse

from util import get_response
from generation.generation import diviner

import ast

PATH_TO_PRETRAINED = '/home/vika/comparative-dialogue/comparative-dialogue/external_pretrained_models/'
MODEL_NAME = 'model2_tagger.hdf5'

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
    return [str_sentence.lower().split() for str_sentence in str_sentences]

def get_objects(list_words, list_tags):
    obj_list = []
    for ind, elem in enumerate(list_tags):
        if elem == 'B-OTHOBJ' or elem == 'B-ASPOBJ':
            obj_list.append(list_words[ind])
    return obj_list

import sys
sys.path.append('/home/vika/NER_RNN/targer')
from src.factories.factory_tagger import TaggerFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simple pipeline')
    parser.add_argument('--input', help='input phrases as a list form, i.e [qwestion1, qwestion2, qwestion3]')
    
    args = parser.parse_args()
    
    x = ast.literal_eval(args.input)
    
    words = create_sequence_from_sentence(['what is better amazon or itunes for showing', 'what is better mouse or rat', 'what is easier to make bread o pizza'])
    model = TaggerFactory.load(PATH_TO_PRETRAINED + MODEL_NAME)
    tags = model.predict_tags_from_words(words)
    
    objects_list = []
    for elem in list(zip(words, tags)):
        objects = get_objects(elem[0], elem[1])
        assert len(objects) >= 2, "We have %d objects to compare" %(len(objects))
        objects_list.append((objects[0], objects[1]))
        
    for obj0, obj1 in objects_list:
        response = get_response(obj0, obj1, False)
        response_json = response.json()
        Merlin = diviner()
        Merlin.create_from_json(response_json)
        Merlin.generate_advice()
        
    print('\nThe end.')