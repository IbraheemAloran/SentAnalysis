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
import pandas as pd


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

pos = open("movieReviews/positive.txt", "r").read()
neg = open("movieReviews/negative.txt", "r").read()


###add each line with label to doc array
doc = []

keyword  = ["J"]
for r in pos.split("\n"):
    doc.append((r, "pos"))

for r in neg.split("\n"):
    doc.append((r, "neg"))

###add each word with label to allWords array
allWords = []
pos_words = nltk.pos_tag(word_tokenize(pos))
neg_words = nltk.pos_tag(word_tokenize(neg))

for w in pos_words:
    if w[1][0] in keyword:
        allWords.append(w[0].lower())

for w in neg_words:
    if w[1][0] in keyword:
        allWords.append(w[0].lower())


print(len(allWords))

allWords = nltk.FreqDist(allWords)
wordfeatures = list(allWords.keys())
#print(wordfeatures)

def find_features(docs):
    # words = set(docs)
    words = word_tokenize(docs)
    features = {}
    for f in wordfeatures:
        features[f] = (f in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in doc]
random.shuffle(featuresets)
#print(len(featuresets))

model = open("FeatureSets/wordfeatures.pickle", "wb")
pickle.dump(wordfeatures, model)
model.close()

model = open("FeatureSets/featuresets.pickle", "wb")
pickle.dump(featuresets, model)
model.close()

model = open("FeatureSets/doc.pickle", "wb")
pickle.dump(doc, model)
model.close()

model = open("FeatureSets/allWords.pickle", "wb")
pickle.dump(allWords, model)
model.close()


classifier = nltk.NaiveBayesClassifier.train(featuresets)

##SKLEARN CLASSIFIERS
mnb = SklearnClassifier(MultinomialNB())
bnb = SklearnClassifier(BernoulliNB())
lr = SklearnClassifier(LogisticRegression())
sgd = SklearnClassifier(SGDClassifier())
svc = SklearnClassifier(SVC())
linsvc = SklearnClassifier(LinearSVC())
nusvc = SklearnClassifier(NuSVC())

###Train sklearn classifiers
mnb.train(featuresets)
print("Multinomial Naive Bayes Complete.")
bnb.train(featuresets)
print("Bernoulli Naive Bayes Complete.")
lr.train(featuresets)
print("Linear Regression Complete.")
sgd.train(featuresets)
print("SGD Complete.")
svc.train(featuresets)
print("SVC Complete.")
linsvc.train(featuresets)
print("Linear SVC Complete.")
nusvc.train(featuresets)
print("NuSVC Complete.")

###VOTE CLASSIFIER
voter = VoteClassifier(mnb,bnb,lr,sgd,svc,linsvc,nusvc)


model = open("Models/naivebayes.pickle", "wb")
pickle.dump(classifier, model)
model.close()

model = open("Models/mnb.pickle", "wb")
pickle.dump(mnb, model)
model.close()

model = open("Models/bnb.pickle", "wb")
pickle.dump(bnb, model)
model.close()

model = open("Models/lr.pickle", "wb")
pickle.dump(lr, model)
model.close()

model = open("Models/sgd.pickle", "wb")
pickle.dump(sgd, model)
model.close()

model = open("Models/svc.pickle", "wb")
pickle.dump(svc, model)
model.close()

model = open("Models/linsvc.pickle", "wb")
pickle.dump(linsvc, model)
model.close()

model = open("Models/nusvc.pickle", "wb")
pickle.dump(nusvc, model)
model.close()