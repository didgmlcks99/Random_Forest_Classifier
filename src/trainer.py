import csv
import normalizer
import modelizer
import timeit
from nltk.tokenize import word_tokenize

# word count then remove by frequency
def count_set(set_words, words_dict):
    
    for word in set_words:

        if word in words_dict.keys():
            words_dict[word] += 1
        else:
            words_dict[word] = 1

def sort_words(words_dict):
    res = {key: val for key, val in sorted(words_dict.items(), key = lambda ele: ele[1], reverse = True)}
    return res

def stopword_rm(words_dict, high_freq, low_freq):

    words_list = list(words_dict.keys())

    for word in words_list:
        if words_dict[word] > high_freq:
            words_dict.pop(word)
        elif words_dict[word] < low_freq:
            words_dict.pop(word)

# predictor model table
def init_predictor(neg_words, non_words):
    model_table = {}

    for word in neg_words:
        if not word in model_table.keys():
            model_table[word] = [0.0, 0.0]
    
    for word in non_words:
        if not word in model_table.keys():
            model_table[word] = [0.0, 0.0]
    
    return model_table

def work_predictor(predictor_model, neg_text, non_text, alpha_num):

    for word in predictor_model:
        neg_count = 0
        non_count = 0

        for text in neg_text:
            if word in text:
                neg_count += 1
        
        for text in non_text:
            if word in text:
                non_count += 1
        
        predictor_model[word][0] = (neg_count+alpha_num) / (len(neg_text)+alpha_num)
        predictor_model[word][1] = (non_count+alpha_num) / (len(non_text)+alpha_num)

def main_train(high_freq, low_freq, alpha_num):
    print("*** running trainer ***")

    neg_text = []
    non_text = []

    neg_word_count = {}
    non_word_count = {}

    prediction_model = {}

    # neg - word count
    with open('../data/train.negative.csv', mode = 'r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            if len(lines) == 0:
                continue
            
            set_words = word_tokenize(lines[0])

            normalizer.normalize_set(set_words)
            count_set(set_words, neg_word_count)

            neg_text.append(set_words)
        
        temp = sort_words(neg_word_count)
        neg_word_count.clear()
        neg_word_count = temp

        stopword_rm(neg_word_count, high_freq, low_freq)

        modelizer.texts_data(neg_text, '../model/train.negative.texts.txt')
        modelizer.count_data(neg_word_count, '../model/train.negative.count.csv')

    # non - word count
    with open('../data/train.non-negative.csv', mode = 'r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            if len(lines) == 0:
                continue
            
            set_words = word_tokenize(lines[0])

            normalizer.normalize_set(set_words)
            count_set(set_words, non_word_count)

            non_text.append(set_words)
        
        temp = sort_words(non_word_count)
        non_word_count.clear()
        non_word_count = temp

        stopword_rm(non_word_count, high_freq, low_freq)

        modelizer.texts_data(non_text, '../model/train.non-negative.texts.txt')
        modelizer.count_data(non_word_count, '../model/train.non-negative.count.csv')


    # make predictor model table (first with words: [neg-perc, non-perc])
    predictor_model = init_predictor(neg_word_count, non_word_count)
    work_predictor(predictor_model, neg_text, non_text, alpha_num)


    # write to predictor model as csv file
    modelizer.model_csv(predictor_model)

    modelizer.print_train_info(neg_text, non_text, high_freq, low_freq, alpha_num, neg_word_count, non_word_count, predictor_model)

def direct_test(analysis_name, val):
    data = [analysis_name, val]
    with open('../analysis/direction.csv', 'w') as file:
        writer = csv.writer(file)

        writer.writerow(data)
    
    directories = analysis_name.split('/')
    splits = directories[2].split('.')
    set_name = splits[0]

    with open(analysis_name, 'w') as file:
        writer = csv.writer(file)

        header = [set_name, 'neg_res', 'non_res']
        writer.writerow(header)



# main
# initialize setting
high_freq = 3000
low_freq = 0
alpha_num = 1

start = timeit.default_timer()

for i in range(high_freq, 0, -20):
    direct_test('../analysis/high-freq', i)
    main_train(i, low_freq, alpha_num)
    exec(open("predictor.py").read())
    print()

for i in range(low_freq, 3000, +20):
    direct_test('../analysis/low-freq', i)
    main_train(high_freq, i, alpha_num)
    exec(open("predictor.py").read())
    print()

power = 1
alph = alpha_num
while alph < 5000:
    alph = 5*power

    direct_test('../analysis/alpha-num', alph)
    
    main_train(high_freq, low_freq, alph)
    exec(open("predictor.py").read())
    power += 1
    print()

stop = timeit.default_timer()
print('Time: ', stop - start)  