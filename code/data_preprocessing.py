import os
import pdb
import sys, re
import cPickle
import argparse
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from collections import Counter, defaultdict
from nltk import ne_chunk, pos_tag, word_tokenize, tokenize

from utils import compute_pos_wemb

neg_liumpqa = pickle.load(open( "../data/lexicon/neg_liumpqa.p", "rb"))
pos_liumpqa = pickle.load(open( "../data/lexicon/pos_liumpqa.p", "rb"))


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def is_not(w):
    '''
    check if a word is 'not'
    '''
    is_not = False
    not_list = ['not', 'NOT', 'Not','never','hardly', 'barely', 'no', 'NO', 'No', ''''t''',\
                'nothing', '''n't''', 'didnt', 'wasnt', 'cant','aint', 'isnt', 'neither',\
                'doesnt', 'wont', 'couldnt', 'cant','dont', 'dnt', 'shouldnt','don', 'none',\
                'nobody', 'nor', 'nowhere', 'rarely', 'scarcely', 'seldom', 'little','cannot']

    if w in not_list or """n't""" in w or """not""" in w:
        is_not = True

    return is_not


def is_turn(w):
    '''
    check if a word is an adversative
    '''
    is_turn = False
    adversatives = ['but', 'still', 'yet', 'whereas', 'while', 'nevertheless', 'rather'\
                    'however', 'despite', 'unless', 'Instead', 'contrast']

    if w in adversatives:
        is_turn = True

    return is_turn


def add_lex_indicator(df):
    '''
    add lexicon indicator to the end of text
    '''
    org_text = df['text']
    text = clean_str(org_text)
    text_tokens = word_tokenize(text)
    lex_indicators = []
    lex_ws = []
    for w in text_tokens:
        if is_not(w):
            lex_indicators.append('4444')
            lex_ws.append(w)
        elif is_turn(w):
            lex_indicators.append('5555')
            lex_ws.append(w)
        elif w.encode('utf-8') in pos_liumpqa:
            lex_indicators.append('1111')
            lex_ws.append(w)
        elif w.encode('utf-8') in neg_liumpqa:
            lex_indicators.append('0000')
            lex_ws.append(w)

    lex_inds = ' '.join(lex_indicators)
    text_lex = text + ' ' + lex_inds
    lex_ws = ' '.join(lex_ws)

    return pd.Series([text_lex, lex_ws])
    