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
doc = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        doc.append((list(movie_reviews.words(fileid)), category))

random.shuffle(doc)

allWords = []
for w in movie_reviews.words():
    allWords.append(w.lower())




allWords = nltk.FreqDist(allWords)
#print(allWords["couples"])
# print(list(allWords.values())[0])
# print(list(allWords.values())[1])
# print(list(allWords.values())[2])
# print(list(allWords.values())[3])

wordfeatures = list(allWords.keys())[:3000]
#print(wordfeatures)

def find_features(docs):
    words = set(docs)
    features = {}
    for f in wordfeatures:
        features[f] = (f in words)

    return features

#print((find_features(movie_reviews.words('neg\cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in doc]

train = featuresets[:1900]
test = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(train)

model = open("naivebayes.pickle", "rb")
classifier = pickle.load(model)
model.close()



print("NLTK Naive Bayes Accuracy: ", (nltk.classify.accuracy(classifier, test)))
classifier.show_most_informative_features(10)

##SKLEARN CLASSIFIERS
mnb = SklearnClassifier(MultinomialNB())
gnb = SklearnClassifier(GaussianNB())
bnb = SklearnClassifier(BernoulliNB())
lr = SklearnClassifier(LogisticRegression())
sgd = SklearnClassifier(SGDClassifier())
svc = SklearnClassifier(SVC())
linsvc = SklearnClassifier(LinearSVC())
nusvc = SklearnClassifier(NuSVC())





###Train sklearn classifiers
mnb.train(train)
#gnb.train(train)
bnb.train(train)
lr.train(train)
sgd.train(train)
svc.train(train)
linsvc.train(train)
nusvc.train(train)


###VOTE CLASSIFIER
voter = VoteClassifier(mnb,bnb,lr,sgd,svc,linsvc,nusvc)


###print accuracies for each model
print("Multinomial Naive Bayes Accuracy: ", (nltk.classify.accuracy(mnb, test)))
#print("Gaussian Naive Bayes Accuracy: ", (nltk.classify.accuracy(gnb, test)))
print("Bernoulli Naive Bayes Accuracy: ", (nltk.classify.accuracy(bnb, test)))
print("Linear Regression Accuracy: ", (nltk.classify.accuracy(lr, test)))
print("SGD Accuracy: ", (nltk.classify.accuracy(sgd, test)))
print("SVC Accuracy: ", (nltk.classify.accuracy(svc, test)))
print("Linear SVC Accuracy: ", (nltk.classify.accuracy(linsvc, test)))
print("NuSVC Accuracy: ", (nltk.classify.accuracy(nusvc, test)))


print("Voter Accuracy: ", (nltk.classify.accuracy(voter, test)))
print("Classification: ", voter.classify(test[0][0]), "Confidence: ", voter.confidence(test[0][0]))
print("Classification: ", voter.classify(test[1][0]), "Confidence: ", voter.confidence(test[1][0]))
print("Classification: ", voter.classify(test[2][0]), "Confidence: ", voter.confidence(test[2][0]))
print("Classification: ", voter.classify(test[3][0]), "Confidence: ", voter.confidence(test[3][0]))
print("Classification: ", voter.classify(test[5][0]), "Confidence: ", voter.confidence(test[5][0]))
print("Classification: ", voter.classify(test[20][0]), "Confidence: ", voter.confidence(test[20][0]))








# model = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, model)
# model.close()



















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



