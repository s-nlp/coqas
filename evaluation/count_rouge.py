from rouge_score import rouge_scorer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import porter
from nltk.tokenize import TweetTokenizer
import collections

stemmer = porter.PorterStemmer()
tokenizer = TweetTokenizer()

def create_ngrams(tokens, n): #сюда добавть эмбединги
    ngrams = collections.Counter()
    ngrams_embs = collections.Counter()
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams

def fmeasure(precision, recall):
  """Computes f-measure given precision and recall values."""

  if precision + recall > 0:
    return 2 * precision * recall / (precision + recall)
  else:
    return 0.0

def cos_sim(emb1, emb2):
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))

def count_ngram_overlap(ngram1_embs, ngram2_embs): #The idea count cousine similarity to every pair. If > 0.6 than add
    result = 1
    #print ("ngram1_embs", len(ngram1_embs), len(ngram1_embs[0]))
    for elem in ngram1_embs: # по словам в н-граме
        similarities = [cos_sim(elem, elem2)[0][0] if cos_sim(elem, elem2) >= 0.6 else 0 for elem2 in ngram2_embs]
        if (len(similarities) == 0):
            return 0.0
        max_ = max(similarities)
        result *= max_
    return result
        

def count_overlap(ngram, ngram_emb, list_of_ngrams2, list_of_ngrams2_embs): # only if ngram not in list_of_ngrams !!!
    overlaps = [count_ngram_overlap(ngram_emb, elem[1]) for elem in list_of_ngrams2_embs] # по всем н-грамам
    if (len(overlaps) == 0):
        return 0.0
    return max(overlaps)

def score_ngrams(target_ngrams, prediction_ngrams):
    intersection_ngrams_count_pred = 0
    intersection_ngrams_count_tg = 0

    
    embeddings_target = bert_embedding([' '.join(elem)for elem in list(target_ngrams.keys())])
    embeddings_predictions = bert_embedding([' '.join(elem)for elem in list(prediction_ngrams.keys())])
    
    for ind, ngram in enumerate(list(target_ngrams.keys())):
        #print ("ind", ind)
        intersection_ngrams_count_tg += min(target_ngrams[ngram],
                                         prediction_ngrams[ngram])
        #print ("min", min(target_ngrams[ngram], prediction_ngrams[ngram]))
        if (min(target_ngrams[ngram], prediction_ngrams[ngram]) == 0):
            #print ("ngram", ngram)
            #print ('overlap')
            overlap_ngram = count_overlap(ngram, embeddings_target[ind][1], prediction_ngrams, embeddings_predictions) #по всем н-грамам
            #print ('overlap ngram', overlap_ngram)
            intersection_ngrams_count_tg += overlap_ngram
            
    for ind, ngram in enumerate(list(prediction_ngrams.keys())):
        #print ("ind", ind)
        intersection_ngrams_count_pred += min(target_ngrams[ngram],
                                         prediction_ngrams[ngram])
        #print ("min", min(target_ngrams[ngram], prediction_ngrams[ngram]))
        if (min(target_ngrams[ngram], prediction_ngrams[ngram]) == 0):
            #print ("ngram", ngram)
            #print ('overlap')
            overlap_ngram = count_overlap(ngram, embeddings_predictions[ind][1], prediction_ngrams, embeddings_target) #по всем н-грамам
            #print ('overlap ngram', overlap_ngram)
            intersection_ngrams_count_pred += overlap_ngram
            
    target_ngrams_count = sum(target_ngrams.values())
    prediction_ngrams_count = sum(prediction_ngrams.values())
    
    precision = intersection_ngrams_count_pred / max(prediction_ngrams_count, 1)
    recall = intersection_ngrams_count_tg / max(target_ngrams_count, 1)
    
    f = fmeasure(precision, recall)
    return f, precision, recall

def score_ngrams_v0(target_ngrams, prediction_ngrams):
    intersection_ngrams_count = 0
    
    #print ("target_ngrams", target_ngrams)
    #print ("prediction_ngrams", prediction_ngrams)
    
    embeddings_target = bert_embedding([' '.join(elem)for elem in list(target_ngrams.keys())])
    embeddings_predictions = bert_embedding([' '.join(elem)for elem in list(prediction_ngrams.keys())])
    
    for ind, ngram in enumerate(list(target_ngrams.keys())):
        #print ("ind", ind)
        intersection_ngrams_count += min(target_ngrams[ngram],
                                         prediction_ngrams[ngram])
        if (min(target_ngrams[ngram], prediction_ngrams[ngram]) == 0):
            #print ("ngram", ngram)
            #print ('overlap')
            overlap_ngram = count_overlap(ngram, embeddings_target[ind][1], prediction_ngrams, embeddings_predictions) #по всем н-грамам
            #print ('overlap ngram', overlap_ngram)
            intersection_ngrams_count += overlap_ngram
            
    target_ngrams_count = sum(target_ngrams.values())
    prediction_ngrams_count = sum(prediction_ngrams.values())
    
    #print ("intersection_ngrams_count", intersection_ngrams_count)
    #print ("intersection_targets_count", prediction_ngrams_count, target_ngrams_count)


    precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
    recall = intersection_ngrams_count / max(target_ngrams_count, 1)
    #print (precision, recall)
    
    f = fmeasure(precision, recall)
    return f, precision, recall

from bert_embedding import BertEmbedding
bert_embedding = BertEmbedding()

def simple_rouge(generated_answers, possible_aswers_list):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3'], use_stemmer=True)
    rouge_1_list = []
    rouge_2_list = []
    rouge_3_list = []
    for ind, elem in enumerate(generated_answers):
        if (elem != "We can't recognize objects for comparision"):
            generated_answer = generated_answers[ind]
            scores = [scorer.score(generated_answer, answ) for answ in possible_aswers_list[ind]]
            sorted_scores_1 = sorted(scores, key=lambda x: x['rouge1'].fmeasure, reverse = True)
            sorted_scores_2 = sorted(scores, key=lambda x: x['rouge2'].fmeasure, reverse = True)
            sorted_scores_3 = sorted(scores, key=lambda x: x['rouge3'].fmeasure, reverse = True)
            rouge_1_list.append(sorted_scores_1[0]['rouge1'])
            rouge_2_list.append(sorted_scores_2[0]['rouge2'])
            rouge_3_list.append(sorted_scores_3[0]['rouge3'])
    return {'rouge1':rouge_1_list, 'rouge2':rouge_2_list, 'rouge3':rouge_3_list}


def rouge_cos3(gen_answers, possible_answers):
    list_of_n1 = []
    list_of_n2 = []
    list_of_n3 = []
    for ind, elem in enumerate(gen_answers):
        print (ind)
        if (elem != "We can't recognize objects for comparision"):
            ngrams1 = create_ngrams(tokenizer.tokenize(elem), 1)
            answ_token_list = [tokenizer.tokenize(elemt) for elemt in possible_answers[0]]
            #scores_list_1 = [score_ngrams(ngrams1, create_ngrams(possible_answ, 1)) for possible_answ in answ_token_list]
            #sorted_scores_1 = sorted(scores_list_1, key=lambda x: x[0], reverse = True)
            #print ("target1", sorted_scores_1[0])
            #list_of_n1.append(sorted_scores_1[0])
            #print (ind, "n2")
            #ngrams01 = create_ngrams(tokenizer.tokenize(elem), 2)
            #scores_list_2 = [score_ngrams(ngrams01, create_ngrams(possible_answ, 2)) for possible_answ in answ_token_list]
            #sorted_scores_2 = sorted(scores_list_2, key=lambda x: x[0], reverse = True)
            #print ("target2", sorted_scores_2[0])
            #list_of_n2.append(sorted_scores_2[0])
            ngrams001 = create_ngrams(tokenizer.tokenize(elem), 3)
            scores_list_3 = [score_ngrams(ngrams001, create_ngrams(possible_answ, 3)) for possible_answ in answ_token_list]
            sorted_scores_3 = sorted(scores_list_3, key=lambda x: x[0], reverse = True)
            print ("target3", sorted_scores_3[0])
            list_of_n3.append(sorted_scores_3[0])
    return {'rouge1':list_of_n1, 'rouge2':list_of_n2, 'rouge3':list_of_n3}

def rouge_cos1(gen_answers, possible_answers):
    list_of_n1 = []
    list_of_n2 = []
    list_of_n3 = []
    for ind, elem in enumerate(gen_answers):
        print (ind)
        if (elem != "We can't recognize objects for comparision"):
            ngrams1 = create_ngrams(tokenizer.tokenize(elem), 1)
            answ_token_list = [tokenizer.tokenize(elemt) for elemt in possible_answers[0]]
            scores_list_1 = [score_ngrams(ngrams1, create_ngrams(possible_answ, 1)) for possible_answ in answ_token_list]
            sorted_scores_1 = sorted(scores_list_1, key=lambda x: x[0], reverse = True)
            print ("target1", sorted_scores_1[0])
            list_of_n1.append(sorted_scores_1[0])
            #print (ind, "n2")
            #ngrams01 = create_ngrams(tokenizer.tokenize(elem), 2)
            #scores_list_2 = [score_ngrams(ngrams01, create_ngrams(possible_answ, 2)) for possible_answ in answ_token_list]
            #sorted_scores_2 = sorted(scores_list_2, key=lambda x: x[0], reverse = True)
            #print ("target2", sorted_scores_2[0])
            #list_of_n2.append(sorted_scores_2[0])
            #ngrams001 = create_ngrams(tokenizer.tokenize(elem), 3)
            #scores_list_3 = [score_ngrams(ngrams001, create_ngrams(possible_answ, 3)) for possible_answ in answ_token_list]
            #sorted_scores_3 = sorted(scores_list_3, key=lambda x: x[0], reverse = True)
            #print ("target3", sorted_scores_3[0])
            #list_of_n3.append(sorted_scores_3[0])
    return {'rouge1':list_of_n1, 'rouge2':list_of_n2, 'rouge3':list_of_n3}


def rouge_cos2(gen_answers, possible_answers):
    list_of_n1 = []
    list_of_n2 = []
    list_of_n3 = []
    for ind, elem in enumerate(gen_answers):
        print (ind)
        if (elem != "We can't recognize objects for comparision"):
            #ngrams1 = create_ngrams(tokenizer.tokenize(elem), 1)
            answ_token_list = [tokenizer.tokenize(elemt) for elemt in possible_answers[0]]
            #scores_list_1 = [score_ngrams(ngrams1, create_ngrams(possible_answ, 1)) for possible_answ in answ_token_list]
            #sorted_scores_1 = sorted(scores_list_1, key=lambda x: x[0], reverse = True)
            #print ("target1", sorted_scores_1[0])
            #list_of_n1.append(sorted_scores_1[0])
            #print (ind, "n2")
            ngrams01 = create_ngrams(tokenizer.tokenize(elem), 2)
            scores_list_2 = [score_ngrams(ngrams01, create_ngrams(possible_answ, 2)) for possible_answ in answ_token_list]
            sorted_scores_2 = sorted(scores_list_2, key=lambda x: x[0], reverse = True)
            print ("target2", sorted_scores_2[0])
            list_of_n2.append(sorted_scores_2[0])
            #ngrams001 = create_ngrams(tokenizer.tokenize(elem), 3)
            #scores_list_3 = [score_ngrams(ngrams001, create_ngrams(possible_answ, 3)) for possible_answ in answ_token_list]
            #sorted_scores_3 = sorted(scores_list_3, key=lambda x: x[0], reverse = True)
            #print ("target3", sorted_scores_3[0])
            #list_of_n3.append(sorted_scores_3[0])
    return {'rouge1':list_of_n1, 'rouge2':list_of_n2, 'rouge3':list_of_n3}

def rouge_cos_all(gen_answers, possible_answers):
    list_of_n1 = []
    list_of_n2 = []
    list_of_n3 = []
    for ind, elem in enumerate(gen_answers):
        print (ind)
        if (elem != "We can't recognize objects for comparision"):
            ngrams1 = create_ngrams(tokenizer.tokenize(elem), 1)
            answ_token_list = [tokenizer.tokenize(elemt) for elemt in possible_answers[0]]
            scores_list_1 = [score_ngrams(ngrams1, create_ngrams(possible_answ, 1)) for possible_answ in answ_token_list]
            sorted_scores_1 = sorted(scores_list_1, key=lambda x: x[0], reverse = True)
            print ("target1", sorted_scores_1[0])
            list_of_n1.append(sorted_scores_1[0])
            print (ind, "n2")
            ngrams01 = create_ngrams(tokenizer.tokenize(elem), 2)
            scores_list_2 = [score_ngrams(ngrams01, create_ngrams(possible_answ, 2)) for possible_answ in answ_token_list]
            sorted_scores_2 = sorted(scores_list_2, key=lambda x: x[0], reverse = True)
            print ("target2", sorted_scores_2[0])
            list_of_n2.append(sorted_scores_2[0])
            ngrams001 = create_ngrams(tokenizer.tokenize(elem), 3)
            scores_list_3 = [score_ngrams(ngrams001, create_ngrams(possible_answ, 3)) for possible_answ in answ_token_list]
            sorted_scores_3 = sorted(scores_list_3, key=lambda x: x[0], reverse = True)
            print ("target3", sorted_scores_3[0])
            list_of_n3.append(sorted_scores_3[0])
    return {'rouge1':list_of_n1, 'rouge2':list_of_n2, 'rouge3':list_of_n3}