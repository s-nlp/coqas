import csv
import pandas as pd
import pickle

from gen import generate_one_answer

import sys
import torch
sys.path.insert(0, "/notebook/cqas")
sys.path.insert(0, "/notebook/cqas/generation/gpt-2-Pytorch")
sys.path.insert(0, "/notebook/cqas/generation/Student")
sys.path.insert(0, "/notebook/cqas/generation/pytorch_transformers")

from generation.generation import diviner
from my_functions import extractor
from my_functions import responser

from cam_summarize import load_cam_model
from text_gen_big import load_big_model
from text_gen import load_small_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LM_CAM = load_cam_model(device)
Cam = diviner(tp = 'cam', model = LM_CAM, device = device)
print ("loaded cam")

from evaluation.gen import generate_one_answer_with_defined_objects

df = pd.DataFrame(columns=['Object 1', 'Object 2', 'Question', 'Best Answer',  'Answers'])




def main():
    df = pd.DataFrame(columns=['Object 1', 'Object 2', 'Question', 'Best Answer',  'Answers'])

    with open('yahoo_answers_positive_questions.csv', 'r') as file:
        reader = csv.reader(file)
        for ind, row in enumerate(reader):
            d = {'Object 1': row[0], 'Object 2': row[1], 'Question': row[2], 'Best Answer': row[3],  'Answers': [elem for elem in row[3:]]}
            if (ind > 0):
                df = df.append(d, ignore_index=True)
            
    for ind, qw in enumerate(df['Question'].values[:500]):
        print (ind, '\n\n')
        obj1 = df['Object 1'][ind]
        obj2 = df['Object 2'][ind]
        print (obj1, obj2)
        answ = generate_one_answer_with_defined_objects(obj1, obj2, Cam)
        print (answ)
        cam_answ_list.append(answ)
        
    with open('cam3_september.pkl', 'wb') as f:
        pickle.dump(cam_answ_list, f)
    
    print ("gpt_noinp vs sent")
                

if __name__ == "__main__":
    main()