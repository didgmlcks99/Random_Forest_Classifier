import csv
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

# no normalization
# no case change
# uppercasing
# lowercasing
# removing non-alphabet chars
# removing non-alphanumeric chars
# not removing word
# removing high frequent word > high_freq
# removing low frequent word < low_freq
# smoothing = alpha_num
def normalize_set(set_words):
    ps = PorterStemmer()

    popped = 0
    for i in range(len(set_words)):
        set_words[i-popped] = lowercase_word(set_words[i-popped])
        set_words[i-popped] = rm_nonalpha(set_words[i-popped])
        set_words[i-popped] = ps.stem(set_words[i-popped])

        if not set_words[i-popped]:
            set_words.pop(i-popped)
            popped+=1

def lowercase_word(word):
    return word.lower()

def uppercase_word(word):
    return word.upper()

def rm_nonalpha(word):
    return ''.join(filter(str.isalpha, word))
    
def rm_nonalnum(word):
    return ''.join(filter(str.isalnum, word))


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


def print_info(neg_size, non_size, neg_text, non_text, high_freq, low_freq, alpha_num, neg_word_count, non_word_count, prediction_model):

    lines = []
    
    lines.append(("stopword frequency upper bound: " + str(high_freq)))
    lines.append(("stopword frequency lower bound: " + str(low_freq)))
    print("stopword frequency upper bound: " + str(high_freq))
    print("stopword frequency lower bound: " + str(low_freq))

    lines.append(("alpha number: " + str(alpha_num)))
    print("alpha number: " + str(alpha_num))

    lines.append(("neg text size: " + str(neg_size)))
    lines.append(("number of words from neg (after normalization and stemming): " + str(len(neg_word_count))))
    print("neg text size: " + str(neg_size))
    print("number of words from neg (after normalization and stemming): " + str(len(neg_word_count)))
    
    lines.append(("non text size: " + str(non_size)))
    lines.append(("number of words from non (after normalization and stemming): " + str(len(non_word_count))))
    print("non text size: " + str(non_size))
    print("number of words from non (after normalization and stemming): " + str(len(non_word_count)))

    lines.append(("total number of words in predictor model: " + str(len(predictor_model))))
    print("total number of words in predictor model: " + str(len(predictor_model)))

    with open('../model/info.txt', 'w') as file:
        file.write('\n'.join(lines))

print("*** running trainer ***")
# main
# initialize setting
neg_size = 0
non_size = 0

neg_text = []
non_text = []

high_freq = 2500
low_freq = 5
alpha_num = 1

neg_word_count = {}
non_word_count = {}

prediction_model = {}

# neg - word count
with open('../data/train.negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        # set_words = lines[0].split()
        set_words = word_tokenize(lines[0])

        normalize_set(set_words)
        count_set(set_words, neg_word_count)

        neg_size += 1

        neg_text.append(set_words)
    
    temp = sort_words(neg_word_count)
    neg_word_count.clear()
    neg_word_count = temp

    stopword_rm(neg_word_count, high_freq, low_freq)

# non - word count
with open('../data/train.non-negative.csv', mode = 'r') as file:
    csvFile = csv.reader(file)

    for lines in csvFile:
        if len(lines) == 0:
            continue
        
        set_words = lines[0].split()

        normalize_set(set_words)
        count_set(set_words, non_word_count)

        non_size += 1

        non_text.append(set_words)
    
    temp = sort_words(non_word_count)
    non_word_count.clear()
    non_word_count = temp

    stopword_rm(non_word_count, high_freq, low_freq)


# make predictor model table (first with words: [neg-perc, non-perc])
predictor_model = init_predictor(neg_word_count, non_word_count)
work_predictor(predictor_model, neg_text, non_text, alpha_num)


# write to predictor model as csv file
with open('../model/predictor-model.csv', 'w') as file:
    writer = csv.writer(file)

    for word in predictor_model:
        line = [word, predictor_model[word][0], predictor_model[word][1]]
        writer.writerow(line)

print_info(neg_size, non_size, neg_text, non_text, high_freq, low_freq, alpha_num, neg_word_count, non_word_count, prediction_model)