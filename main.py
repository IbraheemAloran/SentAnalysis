import nltk

import sentiment_mod as sent
import amazon_sent_mod as amsent
import pandas as pd
from nltk.classify import ClassifierI
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer

education = pd.read_csv("Datasets/Education.csv")
finance = pd.read_csv("Datasets/Finance.csv")
politics = pd.read_csv("Datasets/Politics.csv")
sports = pd.read_csv("Datasets/Sports.csv")


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
        if sent_mod.sentiment(data["Text"][i])[0] == 'pos' and data["Label"][i] == "positive":
            counter = counter + 1
        elif sent_mod.sentiment(data["Text"][i])[0] == 'neg' and data["Label"][i] == "negative":
            counter = counter + 1
        #print(data["label"][i])

    return counter/len(data)


print("Education Accuracy: ", evaluateSentMod(amsent, education))
print("Finance Accuracy: ", evaluateSentMod(amsent, finance))
print("Politics Accuracy: ", evaluateSentMod(amsent, politics))
print("Sports Accuracy: ", evaluateSentMod(amsent, sports))



