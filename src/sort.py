import csv

def count_set(set_words, words_dict):
    
    for word in set_words:

        if word in words_dict.keys():
            words_dict[word] += 1
        else:
            words_dict[word] = 1

def sort_dict(words_dict):
    res = {key: val for key, val in sorted(words_dict.items(), key = lambda ele: ele[1], reverse = True)}
    return res

words_dict = {}

with open('../data/train.negative.csv', mode ='r')as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = lines[0].split()

        count_set(set_words, words_dict)

    temp = sort_dict(words_dict)
    words_dict.clear()
    words_dict = temp

    # print(words_dict)

    x = 0
    for i in words_dict:
        print("{" + i + ": " + str(words_dict[i]) + "}")
        
        x += 1
        if x > 200:
            break