import csv
from nltk.tokenize import word_tokenize

# with open('../data/train.negative.csv', mode = 'r') as file:
#     csvFile = csv.reader(file)

#     i = 0
#     for lines in csvFile:
#         if len(lines) == 0:
#             continue
        
#         # set_words = word_tokenize(lines[0])
#         set_words = lines[0].split()
        
#         print(set_words)

#         if i == 30: break
#         i += 1

list = ['hellomy', 'myheechan']

if 'my' in list:
    print("found")
else:
    print("nope")