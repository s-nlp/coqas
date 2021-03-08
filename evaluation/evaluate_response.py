import csv
import pandas as pd
import pickle

from gen import generate_one_answer

import sys
import torch
from count_rouge import simple_rouge, rouge_cos

def main():
    df = pd.DataFrame(columns=['Object 1', 'Object 2', 'Question', 'Best Answer',  'Answers'])

    with open('yahoo_answers_positive_questions.csv', 'r') as file:
        reader = csv.reader(file)
        for ind, row in enumerate(reader):
            d = {'Object 1': row[0], 'Object 2': row[1], 'Question': row[2], 'Best Answer': row[3],  'Answers': [elem for elem in row[3:]]}
            if (ind > 0):
                df = df.append(d, ignore_index=True)
            
    with open('gpt_answers1.pkl', 'rb') as f:
        gpt_answers = pickle.load(f)
    gpt_scors = rouge_cos(gpt_answers, df['Answers'].values)
    with open('gpt_score.pkl', 'wb') as f:
        pickle.dump(gpt_scors, f)
        
        
    with open('cam_answers1.pkl', 'rb') as f:
        cam_answers = pickle.load(f)
    cam_scors = rouge_cos(cam_answers, df['Answers'].values)
    with open('cam_score.pkl', 'wb') as f:
        pickle.dump(cam_answ, f)
        
    with open('template_answers1.pkl', 'rb') as f:
        template_answers = pickle.load(f)
    templ_scors = rouge_cos(template_answers, df['Answers'].values)
    with open('template_score.pkl', 'wb') as f:
        pickle.dump(templ_scors, f)
                

if __name__ == "__main__":
    main()