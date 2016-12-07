# NBSVM+POS wemb #

NBSVM+POS word embedding (NBSVM+POS wemb) model outperforms most recent published sentiment models 
such CNN, LSTM etc. excepting the model proposed on [Self-Adaptive Hierarchical Sentence Model](https://arxiv.org/abs/1504.05070). 
Comparing with other models, NBSVM+POS wemb is simple and fast-to-train since it’s basically SVM 
using ngram log-count raitos and POS word embedding features.

### NBSVM ###

Naive Bayes Support Vector Machine (NBSVM) is a simple and good approach introduced by [Wang & Manning, 2012](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf). 
This approach computes a log-ratio vector between the average word counts extracted from positive documents 
and the average word counts extracted from negative documents. The input to the logistic regression/SVM classifier 
corresponds to the log-ratio vector multiplied by the binary pattern for each word in the document vector. 
NBSVM often outperforms regular SVM using uni/bi-gram counts directly.

### Improvements of NBSVM+POS wemb###

The performance of NBSVM (originally only use log-ratio vectors for its features) increases about 
0.5-1% by naively incorporating averaged word embeddings into log-ratio feature vectors (NBSVM+avg wemb). 
To push the scores higher, by concatenating averaged word embeddings for different POS tags 
(concatenate several averaged word embedding vectors for nouns, verbs, and adjectives so on) 
instead of the whole sentence to the log-ratio feature vectors (NBSVM + POS wemb), NBSVM + POS wemb 
outperforms NBSVM by 2-3%, and becomes a state-of-the-art model on most of sentiment benchmarks.

####Results of NBSVM+POS###
![Results of NBSVM+POS](../data/results.png?raw=true "Results of NBSVM+POS wemb against other models")
