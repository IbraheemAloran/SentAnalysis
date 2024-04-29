#test methods, funtions and algorithms
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer

text = "Hello Mrs. Doe, hope everything is well. I am excited to learn Python. I am nervous about the project."
stopwrds = set(stopwords.words("english"))
words = word_tokenize(text)
filteredSent = []
trainText = state_union.raw("2005-GWBush.txt")
samplText = state_union.raw("2006-GWBush.txt")

cusTok = PunktSentenceTokenizer(trainText)
tokenz = cusTok.tokenize(samplText)


def tagFile():
    try:
        for i in tokenz:
            words = word_tokenize(i)
            tags = nltk.pos_tag(words)
            print(tags)
    except Exception as e:
        print(str(e))

tagFile()



#ps = PorterStemmer()



#for w in words:
#    if w not in stopwrds:
#        filteredSent.append(w)

#print(sent_tokenize(text))

#print(word_tokenize(text))



