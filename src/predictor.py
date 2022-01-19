import csv
import modelizer
import text_processor
import recorder
import math
from sklearn.preprocessing import StandardScaler

def predict(model, test_neg_text_cases, test_non_text_cases):
    print("> predicting test samples...")
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

    print("> done predicting test samples...")

    return [tp, tn, fp, fn, acc, prec, rec]

def read_test_data(read_fn, tk_case, gram_num, rec_text_fn):
    total_text_word_cases = []

    with open(read_fn, mode = 'r') as file:
        csvFile = csv.reader(file)

        for row in csvFile:
            if len(row) == 0:
                continue
            
            text_word_cases = []
            lined = row[0].splitlines()
            for line in lined:

                text_word_cases = text_processor.setting_tokenizer(line, tk_case)

                text_processor.normalize_set(text_word_cases)
                text_processor.n_gram(text_word_cases, gram_num)
                
                total_text_word_cases.append(text_word_cases)
    
    recorder.record_text_word_cases(total_text_word_cases, rec_text_fn)

    return total_text_word_cases

def get_model(model_fn):
    p_model = {}

    with open(model_fn, mode = 'r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            if len(lines) == 0:
                continue

            p_model[lines[0]] = [float(lines[1]), float(lines[2])]
    
    return p_model

# new RF ================================================================================================================
def test_random_forest(clf, tmp_model):
    # main settings
    gram_num = 2
    tk_case = False

    # case settings
    # test_neg_fn = '../data/test.negative.csv'
    # test_non_fn = '../data/test.non-negative.csv'
    # test_neg_fn = '../data/mytest.negative.csv'
    # test_non_fn = '../data/mytest.non-negative.csv'
    test_neg_fn = '../data/mytrain.negative.csv'
    test_non_fn = '../data/mytrain.non-negative.csv'
    rec_test_neg_fn = '../record/test.negative.texts.txt'
    rec_test_non_fn = '../record/test.non-negative.texts.txt'

    test_neg_text_cases = read_test_data(test_neg_fn, tk_case, gram_num, rec_test_neg_fn)
    test_non_text_cases = read_test_data(test_non_fn, tk_case, gram_num, rec_test_non_fn)

    test_samples = []
    test_samples_classes = []
    
    modelizer.mk_samples(test_samples, test_samples_classes, tmp_model, test_neg_text_cases, 1)
    modelizer.mk_samples(test_samples, test_samples_classes, tmp_model, test_non_text_cases, 0)

    test_samples = StandardScaler().fit(test_samples).transform(test_samples)
    recorder.record_samples(test_samples, test_samples_classes, 'test.samples-model')

    results = clf.predict(test_samples)

    calc_statistics(test_samples_classes, results)

def calc_statistics(test_samples_classes, results):
    print("> predicting test samples...")

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(results)):
        if test_samples_classes[i] == 1 and results[i] == 1:
            tp += 1
        elif test_samples_classes[i] == 0 and results[i] == 1:
            fp += 1
        elif test_samples_classes[i] == 0 and results[i] == 0:
            tn += 1
        elif test_samples_classes[i] == 1 and results[i] == 0:
            fn += 1
    
    acc = (tp + tn) / (tp + fn + tn + fp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    print("> done predicting test samples...")
    
    result = [tp, tn, fp, fn, acc, prec, rec]
    modelizer.print_result_info(result)


# end new RF ============================================================================================================


# # main
# # main settings
# gram_num = 2
# tk_case = False

# # case settings
# test_neg_fn = '../data/test.negative.csv'
# test_non_fn = '../data/test.non-negative.csv'
# rec_test_neg_fn = '../record/test.negative.texts.txt'
# rec_test_non_fn = '../record/test.non-negative.texts.txt'

# model_fn = '../model/predictor-model.csv'

# test_neg_text_cases = read_test_data(test_neg_fn, tk_case, gram_num, rec_test_neg_fn)
# test_non_text_cases = read_test_data(test_non_fn, tk_case, gram_num, rec_test_non_fn)

# prediction_model = get_model(model_fn)
        
# result = predict(prediction_model, test_neg_text_cases, test_non_text_cases)

# modelizer.print_result_info(result)
