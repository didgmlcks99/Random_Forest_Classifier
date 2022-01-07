import csv
import normalizer
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

def print_result(neg_text, non_text, model, result):
    
    lines = []

    lines.append(("number of word in model: " + str(len(model))))
    lines.append(("number of neg text: " + str(len(neg_text))))
    lines.append(("number of non text: " + str(len(non_text))))
    print("number of word in model: " + str(len(model)))
    print("number of neg text: " + str(len(neg_text)))
    print("number of non text: " + str(len(non_text)))

    lines.append(("number of text right for neg: " + str(result[1])))
    lines.append(("percent for neg: " + str(result[0]) + "%"))
    lines.append(("number of text right for neg: " + str(result[2])))
    lines.append(("percent for non: " + str(result[3]) + "%"))
    print("number of text right for neg: " + str(result[1]))
    print("percent for neg: " + str(result[0]) + "%")
    print("number of text right for neg: " + str(result[3]))
    print("percent for non: " + str(result[2]) + "%")

    with open('../model/predictor-result.txt', 'w') as file:
        file.write('\n'.join(lines))


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

with open('../data/test.non-negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = word_tokenize(lines[0])

        normalizer.normalize_set(set_words)

        non_text.append(set_words)

with open('../model/predictor-model.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue

        model[lines[0]] = [float(lines[1]), float(lines[2])]

result = predict(model, neg_text, non_text)

print_result(neg_text, non_text, model, result)