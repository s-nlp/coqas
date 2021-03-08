import csv
import pandas as pd
import pickle

from gen import generate_one_answer, generate_one_answer_with_defined_objects

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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#LM_CAM = load_cam_model(device)
#Cam = diviner(tp = 'cam', model = LM_CAM, device = device)
#print ("loaded cam")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from ctrl_generation import initialize_model
model_type = "ctrl" #PUT NAME OF NEEDED MODEL
length = 200 #MODEL LENGTH

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model, tokenizer, length = initialize_model(model_type, length, device)

CTRL = diviner(tp = 'ctrl', model = model, device = device, tokenizer = tokenizer)
#my_extractor = extractor(my_device = 0)
print ("loaded extractor")


def main():
    df = pd.DataFrame(columns=['Object 1', 'Object 2', 'Question', 'Best Answer',  'Answers'])

    with open('yahoo_answers_positive_questions.csv', 'r') as file:
        reader = csv.reader(file)
        for ind, row in enumerate(reader):
            d = {'Object 1': row[0], 'Object 2': row[1], 'Question': row[2], 'Best Answer': row[3],  'Answers': [elem for elem in row[3:]]}
            if (ind > 0):
                df = df.append(d, ignore_index=True)
            
    gpt_answ_list = []
    for ind, qw in enumerate(df['Question'].values):
        print (ind, '\n\n')
        obj1 = df['Object 1'].values[ind]
        obj2 = df['Object 2'].values[ind]
        answ = generate_one_answer_with_defined_objects(obj1, obj2, CTRL)
        print (answ)
        gpt_answ_list.append(answ)
    #gpt_answ_list = []
    #for ind, qw in enumerate(df['Question'].values):
        #print (ind, '\n\n')
        #answ = generate_one_answer(qw, my_extractor, CTRL)
        #print (answ)
        #gpt_answ_list.append(answ)

    with open('ctrl3_july.pkl', 'wb') as f:
        pickle.dump(gpt_answ_list, f)


if __name__ == "__main__":
    main()