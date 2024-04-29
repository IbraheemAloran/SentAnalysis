#test methods, funtions and algorithms

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

text = "Hello Mrs. Doe, hope everything is well. I am excited to learn Python. I am nervous about the project."
stopwrds = set(stopwords.words("english"))
words = word_tokenize(text)
filteredSent = []

ps = PorterStemmer()



for w in words:
    if w not in stopwrds:
        filteredSent.append(w)

#print(sent_tokenize(text))

#print(word_tokenize(text))



print(filteredSent)