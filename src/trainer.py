import csv
import normalizer
import modelizer
import timeit
import copy
from nltk.tokenize import word_tokenize

# word count then remove by frequency
def count_set(set_words, words_dict):
    
    for word in set_words:

        if word in words_dict.keys():
            words_dict[word] += 1
        else:
            words_dict[word] = 1

def sort_words(words_dict, case):
    res = {key: val for key, val in sorted(words_dict.items(), key = lambda ele: ele[1], reverse = case)}
    return res


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

def work_predictor(predictor_model, train_neg_text_cases, train_non_text_cases, alpha_num):
    print("calculating probablities by bayes theorem")
    for word in predictor_model:
        print(word)
        neg_count = 0
        non_count = 0
        for text in train_neg_text_cases:
            if word in text:
                neg_count += 1
        
        for text in train_non_text_cases:
            if word in text:
                non_count += 1
    
        predictor_model[word][0] = (neg_count+alpha_num) / (len(train_neg_text_cases)+alpha_num)
        predictor_model[word][1] = (non_count+alpha_num) / (len(train_non_text_cases)+alpha_num)
    print("calculating done")

def get_normalized_data(gram_num):
    train_neg_text_cases = []
    train_non_text_cases = []

    neg_word_count = {}
    non_word_count = {}

    # neg - word count
    with open('../data/train.negative.csv', mode = 'r') as file:
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
                count_set(set_words, neg_word_count)

                train_neg_text_cases.append(set_words)

    # non - word count
    with open('../data/train.non-negative.csv', mode = 'r') as file:
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
                count_set(set_words, neg_word_count)

                train_non_text_cases.append(set_words)
    
    modelizer.texts_data(train_neg_text_cases, '../model/train.negative.texts.txt')
    modelizer.texts_data(train_non_text_cases, '../model/train.non-negative.texts.txt')
    
    return [train_neg_text_cases, neg_word_count, train_non_text_cases, non_word_count]

def n_gram(n, set_words):
    grammed_set = []

    if len(set_words) == n-1:
        return set_words
    else:
        for i in range(0, len(set_words)-(n-1), 1):
            word = ''
            
            for j in range(n):
                word += set_words[i+j]

            grammed_set.append(word)
    
    return grammed_set


def fin_lim(predictor_model, high_freq, low_freq, neg_word_count, non_word_count):
    fin_stop(predictor_model, neg_word_count, high_freq, low_freq)
    fin_stop(predictor_model, non_word_count, high_freq, low_freq)

    modelizer.count_data(neg_word_count, '../model/train.negative.count.csv')
    modelizer.count_data(non_word_count, '../model/train.non-negative.count.csv')
    modelizer.model_csv(predictor_model)
    
def fin_stop(predictor_model, words_dict, high_freq, low_freq):

    words_list = list(words_dict.keys())

    for word in words_list:
        if words_dict[word] >= high_freq:
            words_dict.pop(word)
            if word in predictor_model.keys():
                predictor_model.pop(word)
        elif words_dict[word] <= low_freq:
            words_dict.pop(word)
            if word in predictor_model.keys():
                predictor_model.pop(word)

def lim_freq(predictor_model, neg_word_count, non_word_count, case):
    neg_keys = list(neg_word_count.keys())
    non_keys = list(non_word_count.keys())

    neg_bound = 0
    non_bound = 0
    if not neg_word_count:
        neg_bound = -1
    else:
        neg_bound = neg_word_count[neg_keys[0]]
    
    if not non_word_count:
        non_bound = -1
    else:
        non_bound = non_word_count[non_keys[0]]

    bound = get_bound(neg_bound, non_bound, case)

    stopword_rm(predictor_model, neg_word_count, bound)
    stopword_rm(predictor_model, non_word_count, bound)

    modelizer.count_data(neg_word_count, '../model/train.negative.count.csv')
    modelizer.count_data(non_word_count, '../model/train.non-negative.count.csv')
    modelizer.model_csv(predictor_model)

    return bound

def get_bound(neg_bound, non_bound, case):
    bound = 0

    if neg_bound == -1 and non_bound == -1:
        bound = -1
    elif neg_bound == -1:
        bound = non_bound
    elif non_bound == -1:
        bound = neg_bound
    elif case == True:
        if neg_bound >= non_bound:
            bound = neg_bound
        else:
            bound = non_bound
    elif case == False:
        if neg_bound <= non_bound:
            bound = neg_bound
        else:
            bound = non_bound
    
    return bound

def stopword_rm(predictor_model, words_dict, bound):
    words_list = list(words_dict.keys())

    for word in words_list:
        if words_dict[word] == bound:
            words_dict.pop(word)
            if word in predictor_model.keys():
                predictor_model.pop(word)
        else:
            break

def modelize(alpha_num, train_neg_text_cases, train_non_text_cases, neg_word_count, non_word_count):
    # make predictor model table (first with words: [neg-perc, non-perc])
    predictor_model = init_predictor(neg_word_count, non_word_count)
    work_predictor(predictor_model, train_neg_text_cases, train_non_text_cases, alpha_num)

    # write to predictor model as csv file
    modelizer.model_csv(predictor_model)

    return predictor_model

def init_test_analysis(analysis_name):
    directories = analysis_name.split('/')
    splits = directories[3].split('.')
    set_name = splits[0]

    with open(analysis_name, 'w') as file:
        writer = csv.writer(file)

        header = [set_name, 'tp', 'fn', 'fp', 'tn', 'accuracy', 'precision', 'recall']
        writer.writerow(header)

def direct_test(analysis_name, val):
    data = [analysis_name, val]
    with open('../analysis/direction.csv', 'w') as file:
        writer = csv.writer(file)

        writer.writerow(data)


def make_temp_count(neg_word_count, non_word_count, case):
    temp_neg_count = neg_word_count.copy()
    temp_non_count = non_word_count.copy()

    temp = sort_words(temp_neg_count, case)
    temp_neg_count.clear()
    temp_neg_count = temp

    temp = sort_words(temp_non_count, case)
    temp_non_count.clear()
    temp_non_count = temp

    return [temp_neg_count, temp_non_count]



# main
# initialize setting
start = timeit.default_timer()

high_freq = 100000000
low_freq = 0
alpha_num = 1

data = get_normalized_data(2)

train_neg_text_cases = data[0]
train_non_text_cases = data[2]

neg_word_count = data[1]
non_word_count = data[3]

temp = make_temp_count(neg_word_count, non_word_count, True)
neg_word_count = temp[0]
non_word_count = temp[1]
predictor_model = modelize(alpha_num, train_neg_text_cases, train_non_text_cases, neg_word_count, non_word_count)

# train main
name = '../analysis/main/main.csv'
# init_test_analysis(name)

case = True

temp_neg_count = temp[0]
temp_non_count = temp[1]
temp_predictor_model = copy.deepcopy(predictor_model)

upper_bound = 200
lower_bound = 0
alpha_value = 1

fin_lim(temp_predictor_model, upper_bound, lower_bound, temp_neg_count, temp_non_count)
direct_test(name, str(upper_bound)+'/'+str(lower_bound))

modelizer.print_train_info(train_neg_text_cases, train_non_text_cases, upper_bound, lower_bound, alpha_value, temp_neg_count, temp_non_count, temp_predictor_model)
exec(open("predictor.py").read())
print()

stop = timeit.default_timer()
print('Time: ', stop - start)

exit()


# train high freq
name = '../analysis/case5/high-freq.csv'
init_test_analysis(name)

case = True
temp = make_temp_count(neg_word_count, non_word_count, case)
temp_neg_count = temp[0]
temp_non_count = temp[1]

temp_predictor_model = copy.deepcopy(predictor_model)
while True:
    bound = lim_freq(temp_predictor_model, temp_neg_count, temp_non_count, case)
    direct_test(name, bound)
    
    if not temp_predictor_model:
        break
    
    modelizer.print_train_info(train_neg_text_cases, train_non_text_cases, bound, low_freq, alpha_num, temp_neg_count, temp_non_count, temp_predictor_model)
    exec(open("predictor.py").read())
    print()


# train low freq
name = '../analysis/case5/low-freq.csv'
init_test_analysis(name)

case = False
temp = make_temp_count(neg_word_count, non_word_count, case)
temp_neg_count = temp[0]
temp_non_count = temp[1]

temp_predictor_model = copy.deepcopy(predictor_model)
while True:
    bound = lim_freq(temp_predictor_model, temp_neg_count, temp_non_count, case)
    direct_test(name, bound)
    
    if not temp_predictor_model:
        break
    
    modelizer.print_train_info(train_neg_text_cases, train_non_text_cases, high_freq, bound, alpha_num, temp_neg_count, temp_non_count, temp_predictor_model)
    exec(open("predictor.py").read())
    print()

# train alpha num
name = '../analysis/case5/alpha-num.csv'
init_test_analysis(name)

case = True
temp = make_temp_count(neg_word_count, non_word_count, case)
temp_neg_count = temp[0]
temp_non_count = temp[1]

temp_predictor_model = copy.deepcopy(predictor_model)
bound = lim_freq(temp_predictor_model, temp_neg_count, temp_non_count, case)

power = 1
alph = 1
while alph < 100000000000:
    direct_test(name, alph)

    modelizer.print_train_info(train_neg_text_cases, train_non_text_cases, high_freq, low_freq, alph, temp_neg_count, temp_non_count, temp_predictor_model)
    exec(open("predictor.py").read())
    print()

    alph = 2**power
    power += 1

stop = timeit.default_timer()
print('Time: ', stop - start)
