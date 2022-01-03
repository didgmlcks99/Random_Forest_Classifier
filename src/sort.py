import csv

def count_set(word_set, word_dict):
    
    for word in word_set:

        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1

def sorted_tuple(word_dict):
    res = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    return res

word_dict = {}

with open('../data/train.negative.csv', mode ='r')as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
            if len(lines) == 0:
                continue
            
            word_set = lines[0].split()

            count_set(word_set, word_dict)

    x = 0

    sorted_set = sorted_tuple(word_dict)
    # print(sorted_set)

    for i in sorted_set:
        print(i)
        
        x += 1
        if x > 200:
            break