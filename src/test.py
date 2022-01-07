# import re

# str = "hello my name is heechan-yang"

# words = str.split()

# print(words)

# for i in range(words):
#     splitted = re.split('[^a-zA-Z]', words[i])

#     # if len(splitted) > 1:


#     print(splitted)


# importing modules
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

sentence = "Programmers program with programming languages"
words = word_tokenize(sentence)

for w in words:
	print(w, " : ", ps.stem(w))
