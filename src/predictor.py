import csv
import normalizer
import modelizer
import math
from nltk.tokenize import word_tokenize

def predict(model, test_neg_text_cases, test_non_text_cases):
    model_words = list(model.keys())

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # neg
    for text in test_neg_text_cases:
        neg_val = 0.0
        non_val = 0.0

        for word in text:
            if word in model_words:
                neg_val += math.log(model[word][0])
                non_val += math.log(model[word][1])
        
        if neg_val > non_val:
            tp += 1
        else:
            fn += 1
    
    # non
    for text in test_non_text_cases:
        neg_val = 0
        non_val = 0

        for word in text:
            if word in model_words:
                neg_val += math.log(model[word][0])
                non_val += math.log(model[word][1])
        
        if non_val > neg_val:
            tn += 1
        else:
            fp += 1

    acc = (tp + tn) / (tp + fn + tn + fp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    return [tp, tn, fp, fn, acc, prec, rec]



# main
# initialize setting
test_neg_text_cases = []
test_non_text_cases = []
model = {}
gram_num = 2

with open('../data/test.negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = []
        lined = lines[0].splitlines()
        for line in lined:
            # @Usairway hello = ['@', 'Usairway', 'hello']
            # set_words = word_tokenize(line)
            
            # @Usairway hello = ['@Usairway', 'hello']
            set_words = line.split()

            normalizer.normalize_set(set_words)
            normalizer.n_gram(gram_num, set_words)
            
            test_neg_text_cases.append(set_words)
    
modelizer.texts_data(test_neg_text_cases, '../model/test.negative.texts.txt')

with open('../data/test.non-negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = []
        lined = lines[0].splitlines()
        for line in lined:

            # @Usairway hello = ['@', 'Usairway', 'hello']
            # set_words = word_tokenize(line)
            
            # @Usairway hello = ['@Usairway', 'hello']
            set_words = line.split()
            
            normalizer.normalize_set(set_words)
            normalizer.n_gram(gram_num, set_words)
            
            test_non_text_cases.append(set_words)

modelizer.texts_data(test_non_text_cases, '../model/test.non-negative.texts.txt')

with open('../model/predictor-model.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue

        model[lines[0]] = [float(lines[1]), float(lines[2])]
        
result = predict(model, test_neg_text_cases, test_non_text_cases)

modelizer.print_result_info(test_neg_text_cases, test_non_text_cases, model, result)