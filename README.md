
## Dwnloading the Software
* **Using Git** : ```$ git clone https://taoyds@bitbucket.org/taoyds/nbsvm_pos.git```. This requires that you have a bitbucket account and have uploaded your public ssh key.

## Compatibility and Dependencies
Python 2.7 and 3.x
sklearn
scipy
numpy
pandas
nltk

## Running the Model
```
usage: python nbsvm_pos.py --train [path to train in json] --test [path to test in json] --we [path to word2vec] --ngram [e.g. 123]

train/test file should in json with attributes: text: text string, y: labels
```

## Running Example

```
usage: python nbsvm_pos_multiclass.py --train ../data/mr_train_cv2.json --test ../data/mr_test_cv2.json --ngram 123 --we GoogleNews-vectors-negative300.bin

```

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
![Results of NBSVM+POS](raw/master/data/results.png?raw=true "Results of NBSVM+POS wemb against other models")
