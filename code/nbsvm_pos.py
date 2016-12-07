import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from nltk import pos_tag, word_tokenize
from sklearn.datasets import dump_svmlight_file

from utils import load_bin_vec, read_dataset, tokenize, compute_pos_wemb
from data_preprocessing import add_lex_indicator

#code is adapted from https://github.com/mesnilgr/nbsvm

def usage():
    print '#####NBSVM + POS embedding model#####'
    print 'python nbsvm_pos.py --train [path to train in json] --test [path to test in json] --we [path to word2vec] --ngram [e.g. 123]'


def fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid, scoring='f1'):
    '''
    Fit classifier using cross validation

    ---Parameters---

    X, y: training feature matrix, labels
    basemodel: base model e.g. LogisticRegression()
    param_grid: tunning parameters of basemodel
    cv: cross validation folders

    ---Returns---

    best_model: best model return by cross validation

    '''

    # Find the best model within param_grid:
    crossvalidator = GridSearchCV(basemod, param_grid, cv=cv, scoring=scoring)
    crossvalidator.fit(X, y)

    # Report some information:
    print '\n-----------Grid scores-------------'
    for params, mean_score, scores in crossvalidator.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

    print("Best params", crossvalidator.best_params_)
    print("Best score: %0.03f" % crossvalidator.best_score_)
    # Return the best model found:
    best_model = crossvalidator.best_estimator_

    return best_model


def build_counters(df, grams, coln_t='text_lex', coln_y='y'):
    '''
    Build dictionary of tokens for each class in the training data set

    ---Parameters---

    df: training set with attributes y: label, text_lex: text after data preprocessing
    grams : a n-grams rule string (like "123")

    ---Returns---

    counters: the token counter for each class
    '''
    grams = [int(i) for i in grams]
    counters = {}
    count = 0
    for _, rev in df.iterrows():
        c = rev[coln_y]
        text = rev[coln_t]
        count += 1
        # Select class counter
        if c not in counters:
            # we don't have a counter for this class
            counters[c] = Counter()
        counter = counters[c]

        # update counter
        counter.update(tokenize(text, grams))

    print 'number of datum in train: ', count

    return counters


def compute_ratios(counters, alpha=1.0):
    '''
    Calculates the log-count ratios of each token for each class in the training data set

    ---Parameters---

    counters: the token counter for each class
    alpha : smoothing parameter in count vectors (default 1)

    ---Returns---

    dic: a dictionary from token to tokens index
    ratios: log-count ratio
    v: total number of tokens
    '''
    ratios = dict()

    # create a vocabulary - a list of all ngrams
    all_ngrams = set()
    for counter in counters.values():
        all_ngrams.update(counter.keys())
    all_ngrams = list(all_ngrams)
    v = len(all_ngrams)  # the ngram vocabulary size

    # a standard NLP dictionay (ngram -> index map) use to update the
    # one-hot vector p
    dic = dict((t, i) for i, t in enumerate(all_ngrams))

    # sum ngram counts for all classes with alpha smoothing
    # 2* because one gets subtracted when q_c is calculate by subtracting p_c
    sum_counts = np.full(v, 2*alpha)
    for c in counters:
        counter = counters[c]
        for t in all_ngrams:
            sum_counts[dic[t]] += counter[t]

    # calculate r_c for each class
    for c in counters:
        counter = counters[c]
        p_c = np.full(v, alpha)     # initialize p_c with alpha (smoothing)

        # add the ngram counts
        for t in all_ngrams:
            p_c[dic[t]] += counter[t]

        # initialize q_c
        q_c = sum_counts - p_c

        # normalize (l1 norm)
        p_c /= np.linalg.norm(p_c, ord=1)  # = p_c / sum(p_c)
        q_c /= np.linalg.norm(q_c, ord=1)

        # p_c = log(p/|p|)
        p_c = np.log(p_c)
        # q_c = log(not_p/|not_p|)
        q_c = np.log(q_c)

        # Subtract log(not_p/|not_p|
        ratios[c] = p_c - q_c

    return dic, ratios, v


def process_files_ngram(df, dic, r, v, grams, coln_t='text_lex'):
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
    '''

    grams = [int(i) for i in grams]
    n_samples = df.shape[0]
    classes = r.keys()
    X = dict()
    data = dict()
    indptr = [0] # it's a index for the head of doc in indices
    indices = [] # it's a token index list
    for c in classes:
        data[c] = []

    for i, d in df.iterrows():
        text = d[coln_t]
        ngrams = tokenize(text, grams)
        for g in ngrams:
            if g in dic:
                index = dic[g]
                indices.append(index)
                for c in classes:
                    data[c].append(r[c][index])
        indptr.append(len(indices))

    for c in classes:
        X[c] = csr_matrix((data[c], indices, indptr), shape=(n_samples, v), dtype=np.float32)

    if len(classes) == 2:
        log_counts = X[1]
    else:
        log_counts = sp.hstack(X.values(), format='csr')

    return log_counts


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


def model_run(basemod, param_grid, cv, X_train, X_test, y_train, y_test):
    '''
    Running model

    ---Parameters---

    basemodel: base model e.g. LogisticRegression()
    param_grid: tunning parameters of basemodel
    cv: cross validation folders
    train, test: training and testing feature sparse matrices
    y_train, y_test: gold labels for training/testing data set
    coln: column name for adding wembedding (eg: text)

    ---Returns---

    wemb_csr: sparse matrix of POS word embedding features

    '''

    print('training classifiers by CV...')
    model = fit_classifier_with_crossvalidation(X_train, y_train, basemod, cv, param_grid)

    print('testing...')
    y_pred = model.predict(X_test)
    f_score = f1_score(y_test, y_pred)

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

    print "using train to build token count dict for each class..."
    counters = build_counters(train_df, ngram)

    print "computing log-count ratio r..."
    dic, r, v = compute_ratios(counters)

    print 'loading word embedding...'
    word2vec = load_bin_vec(we)

    print "building train and test features --- ngram part..."
    train_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)
    y_train = train_df['y']
    y_test = test_df['y']
    X_train_ngram = process_files_ngram(train_df, dic, r, v, ngram)
    X_test_ngram = process_files_ngram(test_df, dic, r, v, ngram)

    print "building train and test features --- pos embedding part..."
    X_train_embed = process_files_wemb(train_df, word2vec)
    X_test_embed = process_files_wemb(test_df, word2vec)

    print "combining log-count ratio and pos embedding features..."
    X_train = sp.hstack((X_train_ngram, X_train_embed), format='csr')
    X_test = sp.hstack((X_test_ngram, X_test_embed), format='csr')

    print "running model..."
    basemod = LogisticRegression()
    #tunning paramater step especially for multiclass classifier c and class_weight
    cv = 10
    param_grid = [{'C': [1, 0.1], 'class_weight': [{1: 1, -1: 1}, {1: 0.9, -1: 1}, {1: 1, -1: 0.9}]}]
    f_score = model_run(basemod, param_grid, cv, X_train, X_test, y_train, y_test)
    print 'f_score is: ', f_score


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
