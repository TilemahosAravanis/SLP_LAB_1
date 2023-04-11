import glob
import os
import re

import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model

from w2v_train import W2VLossLogger
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = Word2Vec.load("gutenberg_w2v.100d.model")

# NUM_W2V_TO_LOAD = 1000000
# model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True,
# limit=NUM_W2V_TO_LOAD)

SCRIPT_DIRECTORY = os.path.realpath(__file__)

data_dir = "./data/aclImdb/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000


SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)


def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)


def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())


def tokenize(s):
    return s.split(" ")


def preproc_tok(s):
    return tokenize(preprocess(s))


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []
    
    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, "r") as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)

    return data


def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])


def extract_nbow(corpus):
    """Extract neural bag of words representations"""
    nbow_corpus = []
    for comment in corpus:
        nbow = np.zeros((100,))
        for line in comment:
            for word in line.split():
                if word not in model.wv.vocab.keys():
                    continue
                else:
                    nbow = nbow + model.wv[word]
        nbow_corpus.append(nbow)
        
    return np.asarray(nbow_corpus) 


def train_sentiment_analysis(train_corpus, train_labels):
    """Train a sentiment analysis classifier using NBOW + Logistic regression"""
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(train_corpus,train_labels)
    
    return clf


def evaluate_sentiment_analysis(clf, test_corpus, test_labels):
    """Evaluate classifier in the test corpus and report accuracy"""
    y_pred = clf.predict(test_corpus)
    print('/n')
    print("acc of clf: " + str(sklearn.metrics.accuracy_score(test_labels, y_pred)))

if __name__ == "__main__":
    # TODO: read Imdb corpus
    pos = read_samples(pos_train_dir,preprocess) + read_samples(pos_test_dir,preprocess)
    neg = read_samples(neg_train_dir,preprocess) + read_samples(neg_test_dir,preprocess)
    corpus,labels = create_corpus(pos,neg)
    nbow_corpus = extract_nbow(corpus)
    (
        train_corpus,
        test_corpus,
        train_labels,
        test_labels,
    ) = sklearn.model_selection.train_test_split(nbow_corpus, labels, test_size=0.2)
    clf = train_sentiment_analysis(train_corpus, train_labels)
    evaluate_sentiment_analysis(clf, test_corpus, test_labels)
    # TODO: train / evaluate and report accuracy
