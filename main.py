import nltk

import sentiment_mod as sent
import pandas as pd
from nltk.classify import ClassifierI
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer

dataset = pd.read_csv("Datasets/amazon.csv", nrows=100)


# print(dataset["Text"][98])
# print(dataset["label"][98])
#
# print(dataset["Text"][99])
# print(dataset["label"][99])
#
# print(dataset["Text"][97])
# print(dataset["label"][97])
#
# print(sent.sentiment(dataset["Text"][98]))
# print(sent.sentiment(dataset["Text"][99]))
# print(sent.sentiment(dataset["Text"][97]))
#
# print(sent.sentiment("This movie was amazing. I loved the acting and story. the attention to detail was great and the cast was selected perfectly"))
#
# print(sent.sentiment("This movie was not that good. the story and acting were decent. The attention to detail was okay but the cast could have been better"))
#
# print(sent.sentiment("This movie was horrible. the story and acting were decent. The attention to detail was terrible and the cast was awful."))



def evaluateSentMod(sent_mod, data):
    counter = 0
    # print(data["Text"][0])
    # print(data["label"][0])
    #print(sent_mod.sentiment(data["Text"][0])[0] )
    for i in range(len(data)):
        if sent_mod.sentiment(data["Text"][i])[0] == 'pos' and data["label"][i] == 1:
            counter = counter + 1
        elif sent_mod.sentiment(data["Text"][i])[0] == 'neg' and data["label"][i] == 0:
            counter = counter + 1
        #print(data["label"][i])

    return counter/len(data)


#print(evaluateSentMod(sent, dataset))

#print(nltk.classify.accuracy(sent, ))

