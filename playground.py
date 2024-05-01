#test methods, funtions and algorithms
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, abc, movie_reviews, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

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
w1 = wordnet.synsets("ship")[0]
w2 = wordnet.synsets("boat")[0]
print(w1.wup_similarity(w2))

w1 = wordnet.synsets("ship")[0]
w2 = wordnet.synsets("car")[0]
print(w1.wup_similarity(w2))

w1 = wordnet.synsets("dog")[0]
w2 = wordnet.synsets("cat")[0]
print(w1.wup_similarity(w2))

#for w in words:
#    if w not in stopwrds:
#        filteredSent.append(w)

#print(sent_tokenize(text))

#print(word_tokenize(text))



