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

dataset = pd.read_csv("Datasets/amazon.csv")
tagged_words = []
for d in dataset["Text"]:
    tagged_words = tagged_words + nltk.pos_tag(word_tokenize(d))

tagged_words = set(tagged_words)
print("TAGGED WORDS COMPLETE")
###add each line with label to doc array
doc = []

keyword  = ["J"]
for i in range(len(dataset)):
    if dataset["label"][i] == 1:
        doc.append((dataset["Text"][i], "pos"))

    else:
        doc.append((dataset["Text"][i], "neg"))

print("DOC COMPLETE")

###add each word with label to allWords array
allWords = []
for t in tagged_words:
    if t[1][0] in keyword:
        allWords.append(t[0].lower())


allWords = nltk.FreqDist(allWords)
wordfeatures = list(allWords.keys())
#print(wordfeatures)
print("WORD FEATURES COMPLETE")

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

train = featuresets[:16000]
test = featuresets[16000:]


model = open("AmazonFeatures/wordfeatures.pickle", "wb")
pickle.dump(wordfeatures, model)
model.close()

model = open("AmazonFeatures/featuresets.pickle", "wb")
pickle.dump(featuresets, model)
model.close()

model = open("AmazonFeatures/doc.pickle", "wb")
pickle.dump(doc, model)
model.close()

model = open("AmazonFeatures/allWords.pickle", "wb")
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
#nusvc = SklearnClassifier(NuSVC(nu=0.3))

###Train sklearn classifiers
mnb.train(train)
print("Multinomial Naive Bayes Accuracy: ", (nltk.classify.accuracy(mnb, test)))
#gnb.train(train)
bnb.train(train)
print("Bernoulli Naive Bayes Accuracy: ", (nltk.classify.accuracy(bnb, test)))
lr.train(train)
print("Linear Regression Accuracy: ", (nltk.classify.accuracy(lr, test)))
sgd.train(train)
print("SGD Accuracy: ", (nltk.classify.accuracy(sgd, test)))
svc.train(train)
print("SVC Accuracy: ", (nltk.classify.accuracy(svc, test)))
linsvc.train(train)
print("Linear SVC Accuracy: ", (nltk.classify.accuracy(linsvc, test)))
#nusvc.train(train)
#print("NuSVC Accuracy: ", (nltk.classify.accuracy(nusvc, test)))

###VOTE CLASSIFIER
voter = VoteClassifier(mnb,bnb,lr,sgd,svc,linsvc)
print("Voter Accuracy: ", (nltk.classify.accuracy(voter, test)))


model = open("AmazonModels/naivebayes.pickle", "wb")
pickle.dump(classifier, model)
model.close()

model = open("AmazonModels/mnb.pickle", "wb")
pickle.dump(mnb, model)
model.close()

model = open("AmazonModels/bnb.pickle", "wb")
pickle.dump(bnb, model)
model.close()

model = open("AmazonModels/lr.pickle", "wb")
pickle.dump(lr, model)
model.close()

model = open("AmazonModels/sgd.pickle", "wb")
pickle.dump(sgd, model)
model.close()

model = open("AmazonModels/svc.pickle", "wb")
pickle.dump(svc, model)
model.close()

model = open("AmazonModels/linsvc.pickle", "wb")
pickle.dump(linsvc, model)
model.close()

# model = open("AmazonModels/nusvc.pickle", "wb")
# pickle.dump(nusvc, model)
# model.close()