#test methods, funtions and algorithms
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


####TEXT CLASSIFICATION
# doc = []
# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         doc.append((list(movie_reviews.words(fileid)), category))
#
# #random.shuffle(doc)
#
# allWords = []
# for w in movie_reviews.words():
#     allWords.append(w.lower())


# model = open("FeatureSets/doc.pickle", "rb")
# doc = pickle.load(model)
# model.close()
#
# model = open("FeatureSets/allWords.pickle", "rb")
# allWords = pickle.load(model)
# model.close()
#
# model = open("FeatureSets/wordfeatures.pickle", "rb")
# wordfeatures = pickle.load(model)
# model.close()
#
# model = open("FeatureSets/featuresets.pickle", "rb")
# featuresets = pickle.load(model)
# model.close()

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


allWords = nltk.FreqDist(allWords)
#print(allWords["couples"])
# print(list(allWords.values())[0])
# print(list(allWords.values())[1])
# print(list(allWords.values())[2])
# print(list(allWords.values())[3])

wordfeatures = list(allWords.keys())[:5000]
print(wordfeatures)

def find_features(docs):
    # words = set(docs)
    words = word_tokenize(docs)
    features = {}
    for f in wordfeatures:
        features[f] = (f in words)

    return features

#print((find_features(movie_reviews.words('neg\cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in doc]
random.shuffle(featuresets)
print(featuresets[0])
print(featuresets[1])

#positive dataset
train = featuresets[:10000]
test = featuresets[10000:]



# #negative dataset
# train = featuresets[100:]
# test = featuresets[:100]

# model = open("doc.pickle", "wb")
# pickle.dump(doc, model)
# model.close()
#
# model = open("allWords.pickle", "wb")
# pickle.dump(allWords, model)
# model.close()
#
# model = open("wordfeatures.pickle", "wb")
# pickle.dump(wordfeatures, model)
# model.close()
#
# model = open("featuresets.pickle", "wb")
# pickle.dump(featuresets, model)
# model.close()


# classifier = nltk.NaiveBayesClassifier.train(train)

# model = open("Models/naivebayes.pickle", "rb")
# classifier = pickle.load(model)
# model.close()
#
# model = open("Models/mnb.pickle", "rb")
# mnb = pickle.load(model)
# model.close()
#
# model = open("Models/bnb.pickle", "rb")
# bnb = pickle.load(model)
# model.close()
#
# model = open("Models/lr.pickle", "rb")
# lr = pickle.load(model)
# model.close()
#
# model = open("Models/sgd.pickle", "rb")
# sgd = pickle.load(model)
# model.close()
#
# model = open("Models/svc.pickle", "rb")
# svc = pickle.load(model)
# model.close()
#
# model = open("Models/linsvc.pickle", "rb")
# linsvc = pickle.load(model)
# model.close()
#
# model = open("Models/nusvc.pickle", "rb")
# nusvc = pickle.load(model)
# model.close()




#print("NLTK Naive Bayes Accuracy: ", (nltk.classify.accuracy(classifier, test)))
#classifier.show_most_informative_features(10)

##SKLEARN CLASSIFIERS
# mnb = SklearnClassifier(MultinomialNB())
# gnb = SklearnClassifier(GaussianNB())
# bnb = SklearnClassifier(BernoulliNB())
# lr = SklearnClassifier(LogisticRegression())
# sgd = SklearnClassifier(SGDClassifier())
# svc = SklearnClassifier(SVC())
# linsvc = SklearnClassifier(LinearSVC())
# nusvc = SklearnClassifier(NuSVC())





###Train sklearn classifiers
# mnb.train(train)
# print("Multinomial Naive Bayes Accuracy: ", (nltk.classify.accuracy(mnb, test)))
# #gnb.train(train)
# bnb.train(train)
# print("Bernoulli Naive Bayes Accuracy: ", (nltk.classify.accuracy(bnb, test)))
# lr.train(train)
# print("Linear Regression Accuracy: ", (nltk.classify.accuracy(lr, test)))
# sgd.train(train)
# print("SGD Accuracy: ", (nltk.classify.accuracy(sgd, test)))
# svc.train(train)
# print("SVC Accuracy: ", (nltk.classify.accuracy(svc, test)))
# linsvc.train(train)
# print("Linear SVC Accuracy: ", (nltk.classify.accuracy(linsvc, test)))
# nusvc.train(train)
# print("NuSVC Accuracy: ", (nltk.classify.accuracy(nusvc, test)))

###VOTE CLASSIFIER
#voter = VoteClassifier(mnb,bnb,lr,sgd,svc,linsvc,nusvc)


###print accuracies for each model
# print("Multinomial Naive Bayes Accuracy: ", (nltk.classify.accuracy(mnb, test)))
# #print("Gaussian Naive Bayes Accuracy: ", (nltk.classify.accuracy(gnb, test)))
# print("Bernoulli Naive Bayes Accuracy: ", (nltk.classify.accuracy(bnb, test)))
# print("Linear Regression Accuracy: ", (nltk.classify.accuracy(lr, test)))
# print("SGD Accuracy: ", (nltk.classify.accuracy(sgd, test)))
# print("SVC Accuracy: ", (nltk.classify.accuracy(svc, test)))
# print("Linear SVC Accuracy: ", (nltk.classify.accuracy(linsvc, test)))
# print("NuSVC Accuracy: ", (nltk.classify.accuracy(nusvc, test)))
#
#
# print("Voter Accuracy: ", (nltk.classify.accuracy(voter, test)))
# print("Classification: ", voter.classify(test[0][0]), "Confidence: ", voter.confidence(test[0][0]))
# print("Classification: ", voter.classify(test[1][0]), "Confidence: ", voter.confidence(test[1][0]))
# print("Classification: ", voter.classify(test[2][0]), "Confidence: ", voter.confidence(test[2][0]))
# print("Classification: ", voter.classify(test[3][0]), "Confidence: ", voter.confidence(test[3][0]))
# print("Classification: ", voter.classify(test[5][0]), "Confidence: ", voter.confidence(test[5][0]))
# print("Classification: ", voter.classify(test[20][0]), "Confidence: ", voter.confidence(test[20][0]))








# model = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, model)
# model.close()
#
# model = open("mnb.pickle", "wb")
# pickle.dump(mnb, model)
# model.close()
#
# model = open("bnb.pickle", "wb")
# pickle.dump(bnb, model)
# model.close()
#
# model = open("lr.pickle", "wb")
# pickle.dump(lr, model)
# model.close()
#
# model = open("sgd.pickle", "wb")
# pickle.dump(sgd, model)
# model.close()
#
# model = open("svc.pickle", "wb")
# pickle.dump(svc, model)
# model.close()
#
# model = open("linsvc.pickle", "wb")
# pickle.dump(linsvc, model)
# model.close()
#
# model = open("nusvc.pickle", "wb")
# pickle.dump(nusvc, model)
# model.close()


def sentiment(text):
    feat = find_features(text)
    return voter.classify(feat)
















text = "Hello Mrs. Doe, hope everything is well. I am excited to learn Python. I am nervous about the project."
stopwrds = set(stopwords.words("english"))
words = word_tokenize(text)
filteredSent = []
trainText = state_union.raw("2005-GWBush.txt")
samplText = state_union.raw("2006-GWBush.txt")


cusTok = PunktSentenceTokenizer(trainText)
tokenz = cusTok.tokenize(samplText)

###LEMMA
lem = WordNetLemmatizer()
# print(lem.lemmatize("geese"))
# print(lem.lemmatize("mocking", pos="v"))
# print(lem.lemmatize("better", pos="a"))
#
#
# ###TOKEIZE TEXT FILES
# text = abc.raw("science.txt")
# tok = sent_tokenize(text)
# print(tok[5:10])
#
# text = movie_reviews.raw("neg\cv000_29416.txt")
# tok = sent_tokenize(text)
# print(tok[5:10])

def tagFile():
    try:
        for i in tokenz:
            words = word_tokenize(i)
            tags = nltk.pos_tag(words)
            # #chunkParse = nltk.RegexpParser("Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}")
            # chunkParse = nltk.RegexpParser("""Chunk: {<.*>+}
            #                                          }<VB.?|IN|DT|TO>+{""")
            # chunks = chunkParse.parse(tags)
            #
            # chunks.draw()

            ne = nltk.ne_chunk(tags, binary=True)

            ne.draw()





    except Exception as e:
        print(str(e))

#tagFile()



#ps = PorterStemmer()


####SYNONYM SETS
# syn = wordnet.synsets("plan")
# print(syn)
#
# print(syn[0].name())
# print(syn[0].lemmas()[0].name())
# print(syn[0].definition())
# print(syn[0].examples())

syns = []
ants = []


####SYNONYMS AND ANTONYMS
for syn in wordnet.synsets("good"):
    for lem in syn.lemmas():
        # print(lem)
        syns.append(lem.name())
        if lem.antonyms():
            ants.append(lem.antonyms()[0].name())


# print(set(syns))
# print(set(ants))
#####WORD SIMILARITY
# w1 = wordnet.synsets("ship")[0]
# w2 = wordnet.synsets("boat")[0]
# print(w1.wup_similarity(w2))
#
# w1 = wordnet.synsets("ship")[0]
# w2 = wordnet.synsets("car")[0]
# print(w1.wup_similarity(w2))
#
# w1 = wordnet.synsets("dog")[0]
# w2 = wordnet.synsets("cat")[0]
# print(w1.wup_similarity(w2))

#for w in words:
#    if w not in stopwrds:
#        filteredSent.append(w)

#print(sent_tokenize(text))

#print(word_tokenize(text))



