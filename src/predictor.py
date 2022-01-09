import csv
import normalizer
import modelizer
import math
from nltk.tokenize import word_tokenize

def predict(model, test_neg_texts, test_non_texts):
    model_words = list(model.keys())

    neg_true = 0
    non_true = 0

    i = 1
    # neg
    for text in test_neg_texts:
        neg_val = 0.0
        non_val = 0.0

        for word in text:
            if word in model_words:
                neg_val += math.log(model[word][0])
                non_val += math.log(model[word][1])
        
        if neg_val > non_val:
            neg_true += 1
            print("neg " + str(i) + " : correct")
        else:
            print("neg " + str(i) + " : wrong")
        i += 1
    
    i = 1
    # non
    for text in test_non_texts:
        neg_val = 0
        non_val = 0

        for word in text:
            if word in model_words:
                neg_val += math.log(model[word][0])
                non_val += math.log(model[word][1])
        
        if non_val > neg_val:
            non_true += 1
            print("non " + str(i) + " : correct")
        else:
            print("non " + str(i) + " : wrong")
        i += 1
    
    neg_perc = (neg_true / len(test_neg_texts))*100
    non_perc = (non_true / len(test_non_texts))*100

    return [neg_perc, neg_true, non_perc, non_true]



# main
# initialize setting
test_neg_texts = []
test_non_texts = []
model = {}

with open('../data/test.negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = word_tokenize(lines[0])
        normalizer.normalize_set(set_words)
        test_neg_texts.append(set_words)
    modelizer.texts_data(test_neg_texts, '../model/test.negative.texts.txt')

with open('../data/test.non-negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = word_tokenize(lines[0])
        normalizer.normalize_set(set_words)
        test_non_texts.append(set_words)
    modelizer.texts_data(test_non_texts, '../model/test.non-negative.texts.txt')

with open('../model/predictor-model.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue

        model[lines[0]] = [float(lines[1]), float(lines[2])]
        
result = predict(model, test_neg_texts, test_non_texts)

modelizer.print_result_info(test_neg_texts, test_non_texts, model, result)