import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from nltk import pos_tag, word_tokenize
from sklearn.datasets import dump_svmlight_file

from utils import load_bin_vec, read_dataset, tokenize, compute_pos_wemb
from data_preprocessing import add_lex_indicator

def usage():
    print '#####NBSVM + POS embedding model#####'
    print 'python --train [path to train in json] --test [path to test in json] --we [path to word2vec] --ngram [e.g. 123]'


def build_dict(df, grams, coln_t='text_lex', coln_y='y'):
    '''
    Build dictionary of tokens for each class in the training data set

    ---Parameters---

    df: training set with attributes y: label, text_lex: text after data preprocessing
    grams : a n-grams rule string (like "123")

    ---Returns---

    pos_dic, neg_dic: Counter of all tokens among all data for specific class
    '''

    grams = [int(i) for i in grams]
    pos_dic = Counter()
    neg_dic = Counter()
    count = 0
    for _, rev in df.iterrows():
        y = rev[coln_y]
        text = rev[coln_t]
        if y == 1:
            pos_dic.update(tokenize(text, grams))
        else:
            neg_dic.update(tokenize(text, grams))
    return pos_dic, neg_dic


def compute_ratio(poscounts, negcounts, alpha=1):
    '''
    Calculates the log-count ratios for each token in the training data set

    ---Parameters---

    poscounts, negcounts: the token counter of two classes
    alpha : smoothing parameter in count vectors (default 1)

    ---Returns---

    dic: a dictionary from token to tokens index
    r: log-count ratio
    v: total number of tokens
    '''

    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    v = len(alltokens)  # the ngram vocabulary size
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]

    ratio = (abs(p).sum())/float(abs(q).sum())
    q *= ratio
    r = np.log(p/q)
    print "In computing r --- number of tokens: ", str(v)

    return dic, r, v


def process_files_ngram(df, dic, r, v, grams, coln_t='text_lex', coln_y='y'):
    '''
    Process dataframe file to get log-count ngrams ratio vectors
    (feature input for original nbsvm)

    ---Parameters---

    df: the dataframe input file
    dic, r, v: returns from compute_ratio function
    grams: the ngrams rule (like [1,2,3]) (grams)
    coln_t, coln_y: column names for text and label

    ---Returns---

    log_counts: log-count ngrams ratio vectors
    y: label vector
    '''

    grams = [int(i) for i in grams]
    indptr = [0] # it's a index for the head of doc in indices
    indices = [] # it's a token index list
    data = [] # it's a token r list
    n_samples = df.shape[0]
    y = df[coln_y]
    for i, d in df.iterrows():
        l = d[coln_t] # the text
        ngrams = list(set(tokenize(l, grams))) # the tokens array
        for g in ngrams:
            if g in dic:
                index = dic[g]
                indices.append(index)
                data.append(r[index])
        indptr.append(len(indices))
    log_counts = csr_matrix((data, indices, indptr), shape=(n_samples, v), dtype=np.float32) # the ngram part

    return log_counts, y


def process_files_wemb(df, wemb, coln='text'):
    '''
    Process dataframe file to get POS wembedding features in sparse matrix

    ---Parameters---

    df: dataframe file
    wemb: word embedding look up table (eg: word2vec)
    coln: column name for adding wembedding (eg: text)

    ---Returns---

    wemb_csr: sparse matrix of POS word embedding features

    '''

    coeff=[1/3.0, 1/3.0, 1/3.0]#depreciated
    wemb_vec = df[coln].apply((lambda x: compute_pos_wemb(x, wemb, coeff, sep_flag=False)))
    print "In embed process --- the weighted embed matrix shape:", wemb_vec.shape
    wemb_csr = csr_matrix(wemb_vec)

    return wemb_csr


def model_run(basemodel, train, test, y_train, y_test):
    '''
    Running model

    ---Parameters---

    basemodel: base model e.g. LogisticRegression()
    train, test: training and testing feature sparse matrices
    y_train, y_test: gold labels for training/testing data set
    coln: column name for adding wembedding (eg: text)

    ---Returns---

    wemb_csr: sparse matrix of POS word embedding features

    '''

    clf = basemodel
    clf.fit(train, y_train)
    y_pred = clf.predict(test)
    f_score = accuracy_score(y_test, y_pred)

    return f_score

def main(train, test, ngram, we):
    '''
    Given output path (out), liblinear path (liblinear),
    Given ngram string rule (like "123"), ngram
    '''
    global times
    times = 0

    print 'loading data...'
    train_df = read_dataset(train)
    test_df = read_dataset(test)

    print 'cleaning data and add lex indicators...'
    train_df[['text_lex', 'lex_ws']] = train_df.apply(add_lex_indicator, axis=1)
    test_df[['text_lex', 'lex_ws']] = test_df.apply(add_lex_indicator, axis=1)

    print "using train to build pos and neg dic..."
    pos_dic, neg_dic = build_dict(train_df, ngram)

    print "computing log-count ratio r..."
    dic, r, v = compute_ratio(pos_dic, neg_dic)

    print 'loading word embedding...'
    word2vec = load_bin_vec(we)

    print "building train and test features --- ngram part..."
    train_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)
    X_train_ngram, y_train = process_files_ngram(train_df, dic, r, v, ngram)
    X_test_ngram, y_test = process_files_ngram(test_df, dic, r, v, ngram)

    print "building train and test features --- pos embedding part..."
    X_train_embed = process_files_wemb(train_df, word2vec)
    X_test_embed = process_files_wemb(test_df, word2vec)

    print "combining log-count ratio and pos embedding features..."
    train_f = sp.hstack((X_train_ngram, X_train_embed), format='csr')
    test_f = sp.hstack((X_test_ngram, X_test_embed), format='csr')

    print "running model..."
    basemodel = LogisticRegression()
    f_score = model_run(basemodel, train_f, test_f, y_train, y_test)
    print '##############f_score is: ', f_score

    print 'model ended.'


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Run NB-SVM on some text files.')
        parser.add_argument('--train', help='path to train data set in json (contains text and y columns)')
        parser.add_argument('--test', help='path to test data set in json')
        parser.add_argument('--ngram', help='N-grams considered e.g. 123 is uni+bi+tri-grams')
        parser.add_argument('--we', help='path to word embedding e.g. word2vec')
        args = vars(parser.parse_args())
    except:
        usage()

    main(**args)
