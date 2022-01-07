import csv
import normalizer
import modelizer
import math
from nltk.tokenize import word_tokenize

def predict(model, neg_text, non_text):
    model_words = list(model.keys())

    neg_true = 0
    non_true = 0

    for text in neg_text:
        neg_val = 0.0
        non_val = 0.0

        for word in text:
            if word in model_words:
                neg_val += math.log(model[word][0])
                non_val += math.log(model[word][1])
        
        if neg_val > non_val:
            neg_true += 1
    
    for text in non_text:
        neg_val = 0
        non_val = 0

        for word in text:
            if word in model_words:
                neg_val += math.log(model[word][0])
                non_val += math.log(model[word][1])
        
        if non_val > neg_val:
            non_true += 1
    
    neg_perc = (neg_true / len(neg_text))*100
    non_perc = (non_true / len(non_text))*100

    return [neg_perc, neg_true, non_perc, non_true]


print("*** running predictor ***")
# main
# initialize setting
neg_text = []
non_text = []
model = {}

with open('../data/test.negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = word_tokenize(lines[0])

        normalizer.normalize_set(set_words)

        neg_text.append(set_words)
    
    modelizer.texts_data(neg_text, '../model/test.negative.texts.txt')

with open('../data/test.non-negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = word_tokenize(lines[0])

        normalizer.normalize_set(set_words)

        non_text.append(set_words)
    
    modelizer.texts_data(non_text, '../model/test.non-negative.texts.txt')

with open('../model/predictor-model.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue

        model[lines[0]] = [float(lines[1]), float(lines[2])]

result = predict(model, neg_text, non_text)

modelizer.print_result_info(neg_text, non_text, model, result)