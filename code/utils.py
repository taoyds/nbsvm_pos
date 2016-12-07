__author__ = "Tao Yu"
__version__ = "Dec. 07 2016"

import os
import sys
import csv
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from nltk import ne_chunk, pos_tag, word_tokenize, tokenize


def tokenize(sentence, grams):
    '''
    Calculates the log-count ratios for each token in the training data set

    ---Parameters---

    sentence: a sentence string
    grams : a n-grams rule string (like "123")

    ---Returns---

    tokens: all possible ngram tokens in the sentence
    '''

    words = word_tokenize(sentence)
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]

    return tokens


def load_bin_vec(fname):
    '''
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    '''
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')

    return word_vecs


def read_dataset(data_path):
    '''
    Read the dataset in JSON into pandas Dataframe
    '''
    if os.path.isfile(data_path):
        try:
            data = pd.read_json(data_path)
            return data
        except:
            print('Could not read dataset into pandas Dataframe. Input data has to be in JSON.')
    else:
        print('Could not find the data file.')


def compute_pos_wemb(text, wemb, coeffs, sep_flag=False):
    '''
    Add word embedding features based on POS tags
    '''
    '''
    Add word embedding features based on POS tags

    ---Parameters---

    text: text to be tokenized
    wemb: word embedding look up table (eg: word2vec)
    ind: word embedding feature start index
    coeffs: coefficients for nouns, adjectives and verbs (depreciated)
    sep_flag: if treat all word the same
    is_concat: if concatenate word embedding vectors for nouns, adjectives and verbs

    ---Returns---

    wemb_vec: POS word embedding features
    '''

    try:
        word_tokens = word_tokenize(text)
        if sep_flag:
            words_w_tag = []
            for w in word_tokens:
                words_w_tag.append((w, 'NN'))
        else:
            words_w_tag = pos_tag(word_tokens)
    except:
        print 'Text input should not be tokenized!!!!!!!'


    wemb_dim = len(wemb['summer'])
    backer = np.zeros(wemb_dim)
    tag_word2vec = {'nouns': [], 'verbs': [], 'adjs': []}
    nouns = ['NNS', 'NN','NNP', 'NNPS']
    verbs = ['VBP', 'VBZ', 'MD', 'VBG', 'VBD', 'VB', 'VBN', 'IN']
    adjs = ['RB', 'RBR', 'RBS', 'JJR', 'JJ', 'JJS']

    for w_tag in words_w_tag:
        w = w_tag[0]
        tag = w_tag[1]
        try:
            if tag in nouns:
                tag_word2vec['nouns'].append(wemb[w])
            elif tag in verbs:
                tag_word2vec['verbs'].append(wemb[w])
            elif tag in adjs:
                tag_word2vec['adjs'].append(wemb[w])
        except Exception:
            pass

    zero_cfs = 0
    for tag, w2vs in tag_word2vec.items():
        if len(w2vs) == 0:
            zero_cfs += 1
            if tag == 'nouns':
                coeffs[0] = 0
                tag_word2vec[tag] = backer
            elif tag == 'verbs':
                coeffs[1] = 0
                tag_word2vec[tag] = backer
            elif tag == 'adjs':
                coeffs[2] = 0
                tag_word2vec[tag] = backer

    if zero_cfs == 1:
        for i, cf in enumerate(coeffs):
            if cf != 0:
                coeffs[i] = 0.5
    elif zero_cfs == 2:
        for i, cf in enumerate(coeffs):
            if cf != 0:
                coeffs[i] = 1

    tag_avgword2vec = {}
    for tag, w2vs in tag_word2vec.items():
        num_w2v = float(len(w2vs))
        total_w2v = np.zeros(wemb_dim)
        for w2v in w2vs:
            total_w2v += w2v
        avg_w2v = total_w2v / num_w2v
        tag_avgword2vec[tag] = avg_w2v

    keywords2vec = np.zeros(wemb_dim)
    for tag, avgw2vs in tag_avgword2vec.items():
        if tag == 'nouns':
            keywords2vec += coeffs[0] * tag_avgword2vec[tag]
        elif tag == 'verbs':
            keywords2vec += coeffs[1] * tag_avgword2vec[tag]
        elif tag == 'adjs':
            keywords2vec += coeffs[2] * tag_avgword2vec[tag]

    keywords2vec = np.concatenate((tag_avgword2vec['nouns'],\
                    tag_avgword2vec['verbs'], tag_avgword2vec['adjs']), axis=0)
    wemb_vec = pd.Series(keywords2vec)

    return wemb_vec
