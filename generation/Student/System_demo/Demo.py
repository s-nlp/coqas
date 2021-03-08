from utils.objects import Argument
import requests
from requests.auth import HTTPBasicAuth
from utils.sentence_clearer import clear_sentences, remove_questions
from ml_approach.sentence_preparation_ML import prepare_sentence_DF
from ml_approach.classify import classify_sentences
from utils.es_requester import extract_sentences
from pke.unsupervised import MultipartiteRank
import gensim
import numpy as np
import pandas as pd
import gensim.downloader as api
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from template_generation.template_generation import generate_template
from gensim.summarization.textcleaner import split_sentences
from gensim.summarization.summarizer import summarize



def one_liner(obj_a, obj_b, user, password, w2v_model=None):
    obj_a = Argument(obj_a.lower().strip())
    obj_b = Argument(obj_b.lower().strip())
    print("Requesting Elasticsearch")
    json_compl = request_elasticsearch(obj_a, obj_b, 'reader', 'reader')

    print("Preparing sentences")

    all_sentences = extract_sentences(json_compl)
    remove_questions(all_sentences)
    prepared_sentences = prepare_sentence_DF(all_sentences, obj_a, obj_b)

    print("Classifying comparative sentences")

    classification_results = classify_sentences(prepared_sentences, 'bow')

    comparative_sentences = prepared_sentences[classification_results['max'] != 'NONE']
    comparative_sentences['max'] = classification_results[classification_results['max'] != 'NONE']['max']

    print("Looking for keyphrases")

    text = prepared_sentences[classification_results['max'] != 'NONE']['sentence'].str.cat(sep=' ')

    extractor = MultipartiteRank()
    extractor.load_document(input=text, language="en", normalization='stemming')

    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})

    extractor.candidate_weighting()

    keyphrases = extractor.get_n_best(n=-1, stemming=False)


    if w2v_model is None:
        print("Loading w2v")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # w2v_model = api.load('word2vec-google-news-300')

    print("Preparing keyphrases for classification")

    asp_df = pd.DataFrame(columns=['OBJECT A', 'OBJECT B', 'ASPECT', 'SENTENCE', 'max'])
    forbidden_phrases = [obj_a.name, obj_b.name, 'better', 'worse']

    for index, row in comparative_sentences.iterrows():
        sentence = row['sentence']
        for (keyphrase, score) in keyphrases:
            skip_keyphrase = False
            for phrase in forbidden_phrases:
                if keyphrase == phrase:
                    skip_keyphrase = True
                    break
            if not skip_keyphrase:
                if keyphrase in sentence:
                    asp_df = asp_df.append(
                        {'OBJECT A': row['object_a'],
                         'OBJECT B': row['object_b'],
                         'ASPECT': keyphrase,
                         'SENTENCE': row['sentence'],
                         'max': row['max'],
                         }, ignore_index=True)

    asp_df['TOKENS'] = pd.Series(get_list_of_tokens(asp_df))
    X_asp = to_w2v_matrix(asp_df, w2v_model)

    print("Classifying keyphrases")

    filename = 'asp_clf.pkl'
    model = pickle.load(open(filename, 'rb'))

    y_pred = model.predict(X_asp)
    aspects = asp_df.iloc[np.nonzero(y_pred)[0].tolist()]['ASPECT'].unique()

    print("Determining the winner")

    obj_a_aspects = []
    obj_b_aspects = []
    for aspect in aspects:
        rows = asp_df[asp_df['ASPECT'] == aspect]
        if obj_a.name == rows.iloc[0]['OBJECT A']:
            obj_a_aspects.append(aspect)
        else:
            obj_b_aspects.append(aspect)

    comparing_pair = {}
    if len(obj_a_aspects) > len(obj_b_aspects):
        comparing_pair['winner_aspects'] = obj_a_aspects
        comparing_pair['loser_aspects'] = obj_b_aspects
        comparing_pair['winner'] = obj_a.name
        comparing_pair['loser'] = obj_b.name
    else:
        comparing_pair['winner_aspects'] = obj_b_aspects
        comparing_pair['loser_aspects'] = obj_a_aspects
        comparing_pair['winner'] = obj_b.name
        comparing_pair['loser'] = obj_a.name

    print("Generating response")

    response = generate_template(comparing_pair, mode="extended")

    print("Generating summary")

    rows = asp_df[asp_df.ASPECT.isin(aspects)]

    sentences = ""
    for row in range(rows.shape[0]):
        sentence = asp_df.iloc[row]['SENTENCE'] + " "
        if sentence not in sentences:
            sentences += sentence

    summary = ""

    if len(split_sentences(sentences)) > 10:
        summary = str(summarize(sentences, split=False, word_count=50))
    else:
        summary = sentences

    return response, summary



def request_elasticsearch(obj_a, obj_b, user, password):
    url = 'http://ltdemos.informatik.uni-hamburg.de/depcc-index/_search?q='
    url += 'text:\"{}\"%20AND%20\"{}\"'.format(obj_a.name, obj_b.name)

    size = 10000

    url += '&from=0&size={}'.format(size)
    response = requests.get(url, auth=HTTPBasicAuth(user, password))
    return response

def create_sentence_embeddings(model, words_list):
    sentence_embedding = []
    for word in words_list:
        try:
            sentence_embedding.append(model[word])
        except KeyError:
            continue
#             print(word + " is not in the vocabulary, skipping...")
    if len(sentence_embedding) == 0:
        sentence_embedding.append(np.zeros(300))
    return np.array(sentence_embedding)

def to_w2v_matrix(df_data, model):
    sent_embs = np.zeros([df_data.shape[0], 300 * 4], dtype='float32')
    for i in range(df_data.shape[0]):
        object_a_embedding = create_sentence_embeddings(model, df_data["OBJECT A"][i].split()).mean(axis=0)
        object_b_embedding = create_sentence_embeddings(model, df_data["OBJECT B"][i].split()).mean(axis=0)
        aspect_embedding = create_sentence_embeddings(model, df_data["ASPECT"][i].split()).mean(axis=0)
        sentence_embedding = create_sentence_embeddings(model, df_data["TOKENS"][i]).mean(axis=0)
        sent_embs[i, :] = np.concatenate((object_a_embedding, object_b_embedding, aspect_embedding, sentence_embedding), axis=0)
    return sent_embs

def get_list_of_tokens(df_texts):
    stop_words=set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = []
    texts = df_texts["SENTENCE"].values
    for i in range(len(texts)):
        row = texts[i]
        # remove punctuation
        for ch in string.punctuation:
            row = row.replace(ch, " ")
        row = row.replace("   ", " ")
        row = row.replace("  ", " ")
        temp_line = []
        # remove stop words
        for word in row.split():
            if word not in stop_words:
                temp_line.append(word)
        row = ' '.join(temp_line)
        # lemmatization
        temp_line = []
        for word in row.split():
            temp_line.append(wordnet_lemmatizer.lemmatize(word))
        tokens.append(temp_line)
    return tokens