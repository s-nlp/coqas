import csv
import pandas as pd
import pickle

from gen import generate_one_answer

import sys
import torch
from count_rouge import simple_rouge, rouge_cos1

def main():
    df = pd.DataFrame(columns=['Object 1', 'Object 2', 'Question', 'Best Answer',  'Answers'])

    with open('yahoo_answers_positive_questions.csv', 'r') as file:
        reader = csv.reader(file)
        for ind, row in enumerate(reader):
            d = {'Object 1': row[0], 'Object 2': row[1], 'Question': row[2], 'Best Answer': row[3],  'Answers': [elem for elem in row[3:]]}
            if (ind > 0):
                df = df.append(d, ignore_index=True)
            
    with open('templ_answers1.pkl', 'rb') as f:
        template_answers = pickle.load(f)
    templ_scors = rouge_cos1(template_answers[:200], df['Answers'].values[:200])
    with open('template_score1_.pkl', 'wb') as f:
        pickle.dump(templ_scors, f)
                

if __name__ == "__main__":
    main()