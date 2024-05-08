import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, abc, movie_reviews, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice = votes.count(mode(votes))
        conf = choice / len(votes)
        return conf


model = open("AmazonFeatures/doc.pickle", "rb")
doc = pickle.load(model)
model.close()

model = open("AmazonFeatures/allWords.pickle", "rb")
allWords = pickle.load(model)
model.close()

model = open("AmazonFeatures/wordfeatures.pickle", "rb")
wordfeatures = pickle.load(model)
model.close()

model = open("AmazonFeatures/featuresets.pickle", "rb")
AmazonFeatures = pickle.load(model)
model.close()


def find_features(docs):
    # words = set(docs)
    words = word_tokenize(docs)
    features = {}
    for f in wordfeatures:
        features[f] = (f in words)

    return features


model = open("AmazonModels/naivebayes.pickle", "rb")
classifier = pickle.load(model)
model.close()

model = open("AmazonModels/mnb.pickle", "rb")
mnb = pickle.load(model)
model.close()

model = open("AmazonModels/bnb.pickle", "rb")
bnb = pickle.load(model)
model.close()

model = open("AmazonModels/lr.pickle", "rb")
lr = pickle.load(model)
model.close()

model = open("AmazonModels/sgd.pickle", "rb")
sgd = pickle.load(model)
model.close()

model = open("AmazonModels/svc.pickle", "rb")
svc = pickle.load(model)
model.close()

model = open("AmazonModels/linsvc.pickle", "rb")
linsvc = pickle.load(model)
model.close()

voter = VoteClassifier(mnb,bnb,lr,sgd,svc,linsvc)


def sentiment(text):
    feat = find_features(text)
    return voter.classify(feat), voter.confidence(feat)


