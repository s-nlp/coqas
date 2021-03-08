import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_pretrained_bert import *
import matplotlib.pyplot as plt

import os
current_directory_path = os.path.dirname(os.path.realpath(__file__))


from tokenizer_custom_bert import BertTokenizer

import networkx as nx

def init_pagerank(scores):
    weights = {}
    n_edges = {}
    for i, j, w in scores:
        prev_score, prev_n = weights.get(i, 0.0), n_edges.get(i, 0)
        weights[i] = w + prev_score
        n_edges[i] = 1 + prev_n
        prev_score, prev_n = weights.get(j, 0.0), n_edges.get(j, 0)
        weights[j] = w + prev_score
        n_edges[j] = 1 + prev_n
    for k in weights:
        v = weights[k]
        weights[k] = v/n_edges[k]
    return weights


def extract_top(scores, k = 10, weighted_init = False):
    g = nx.Graph()
    g.add_weighted_edges_from(scores)
    if weighted_init:
        pr = nx.pagerank(g, nstart = init_pagerank(scores))
    else:
        pr = nx.pagerank(g)
    idxes = sorted(pr, key = lambda x: -pr[x])[:k]
    return idxes

def write_sentences(sample, sentences = None):
    if sentences is None:
        sentences_obj1 = []
        sentences_obj2 = []
    for s in sample['object1']['sentences']:
        sentences_obj1.append(s['text'])
    for s in sample['object2']['sentences']:
        sentences_obj2.append(s['text'])
    return sentences_obj1, sentences_obj2

def similarity(s1, s2):
    return s1.dot(s2)/np.linalg.norm(s1,2)/np.linalg.norm(s2, 2)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp = 768, nhid = 512, nlayers=1, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear( nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, input_lengths = None):
        emb = self.drop(self.encoder(input))
        if input_lengths is not None:
            emb = torch.nn.utils.rnn.pack_padded_sequence(emb, input_lengths, enforce_sorted = True)
            outputs, hidden = self.rnn(emb, hidden)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        else:
            output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
        

def detach_hidden(hidden):
    return (hidden[0].detach(),
            hidden[1].detach())

def prune_graph(scores):
    mean = np.array([s[2] for s in scores]).mean()
    pruned = []
    for s in scores:
        if s[2] > mean:
            pruned.append(s)
    return pruned

def load_cam_model(device):
    try:
        print (11, device)
        LM = RNNModel(30522, nlayers=2, dropout = 0.0)
        print (11)
        LM.load_state_dict(torch.load(current_directory_path + '/wikitext_lm_finetuned', map_location=lambda storage, loc: storage))
        print (12)
        LM = LM.to(device)
    except:
        raise RuntimeError("Can't map CAM to gpu. Maybe it is OOM")
    return LM

def create_sumaries(LM, sentences, raw_sentences, device):
    k = 5
    prune = True
    exclude_common = True
    weighted_init = True
    summaries = []
    hiddens = []
    for s in sentences:
        hidden = LM.init_hidden(1)
        batch = torch.LongTensor([s]).transpose(0, 1).to(device)
        preds, h = LM(batch, hidden, input_lengths = None)
        hiddens.append(h[1].view(-1).cpu().detach().numpy())

    scores = []
    print (14)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            scores.append((i, j, similarity(hiddens[i], hiddens[j])))

    if prune:
        scores = prune_graph(scores)

    top_k = extract_top(scores, k = k, weighted_init = weighted_init)

    summaries.append([raw_sentences[i] for i in top_k])
    return (' '.join(summaries[0]))


def cam_summarize(input_json, LM, device):
    print ("cam_summarize")
    #print (input_json)
    print (current_directory_path + '/vocab.txt')
    tokenizer = BertTokenizer(vocab_file = current_directory_path + '/vocab.txt')
    print (11)
    raw_sentences_obj1, raw_sentences_obj2 = write_sentences(input_json)
    summaries = []
    sentences_obj1 = []
    sentences_obj2 = []
    for s in raw_sentences_obj1:
        s = ["[CLS]"] + tokenizer.tokenize(s) + ["[SEP]"]
        sentences_obj1.append(tokenizer.convert_tokens_to_ids(s))
    for s in raw_sentences_obj2:
        s = ["[CLS]"] + tokenizer.tokenize(s) + ["[SEP]"]
        sentences_obj2.append(tokenizer.convert_tokens_to_ids(s))
    print (13)
    summaries_1 = create_sumaries(LM, sentences_obj1, raw_sentences_obj1, device)
    print ("summaries 1 ", summaries_1, '\n')
    summaries_2 = create_sumaries(LM, sentences_obj2, raw_sentences_obj2, device)
    print ("summaries 2 ", summaries_2, '\n')
    full_text = ''
    print (input_json['winner'] == input_json['object1']['name'])
    print (input_json['winner'], input_json['object1']['name'])
    if (input_json['winner'] == input_json['object1']['name']):
        full_text = summaries_1 + '\n' + summaries_2
    else:
        full_text = summaries_2 + '\n' + summaries_1
    print ("full text in cam summarize", full_text)
    return full_text