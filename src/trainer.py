import csv
import text_processor
import recorder
import modelizer
import timeit
import copy

def read_train_data(read_fn, tk_case, gram_num, rec_text_fn):
    total_text_word_cases = []
    word_cases_count_dict = {}

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
                modelizer.count_text_word_cases(text_word_cases, word_cases_count_dict)

                total_text_word_cases.append(text_word_cases)
    
    recorder.record_text_word_cases(total_text_word_cases, rec_text_fn)
    
    return [total_text_word_cases, word_cases_count_dict]


# main
# initialize setting
start = timeit.default_timer()

# main settings
tk_case = False
default_sort_order = True
high_sort_order = True
low_sort_order = False
run_case = False

# model settings
gram_num = 2
high_freq = 100000000000
low_freq = 0
alpha_num = 1

# case settings
train_neg_fn = '../data/train.negative.csv'
train_non_fn = '../data/train.non-negative.csv'
rec_train_neg_fn = '../record/train.negative.texts.txt'
rec_train_non_fn = '../record/train.non-negative.texts.txt'

tmp = read_train_data(train_neg_fn, tk_case, gram_num, rec_train_neg_fn)
train_neg_texts = tmp[0]
main_neg_cases_count_dict = modelizer.sort_word_cases(tmp[1], default_sort_order)

tmp = read_train_data(train_non_fn, tk_case, gram_num, rec_train_non_fn)
train_non_texts = tmp[0]
main_non_cases_count_dict = modelizer.sort_word_cases(tmp[1], default_sort_order)

main_model = modelizer.mk_model(alpha_num, train_neg_texts, train_non_texts, main_neg_cases_count_dict, main_non_cases_count_dict)

if run_case == True:
    # start main train
    tmp_neg_count_dict = modelizer.make_tmp_data(main_neg_cases_count_dict, default_sort_order)
    tmp_non_count_dict = modelizer.make_tmp_data(main_non_cases_count_dict, default_sort_order)
    tmp_model = copy.deepcopy(main_model)

    name = '../analysis/main/main.csv'
    # recorder.init_test_analysis(name)
    
    modelizer.finalize_model(tmp_model, high_freq, low_freq, tmp_neg_count_dict, tmp_non_count_dict, run_case, default_sort_order)
    recorder.direct_test(name, str(high_freq)+'/'+str(low_freq))

    exec(open("predictor.py").read())
    print()
else:
    # train high freq
    name = '../analysis/case6/high-freq.csv'
    recorder.init_test_analysis(name)

    tmp_neg_count_dict = modelizer.make_tmp_data(main_neg_cases_count_dict, high_sort_order)
    tmp_non_count_dict = modelizer.make_tmp_data(main_non_cases_count_dict, high_sort_order)
    tmp_model = copy.deepcopy(main_model)

    while True:
        bound = modelizer.finalize_model(tmp_model, high_freq, low_freq, tmp_neg_count_dict, tmp_non_count_dict, run_case, high_sort_order)
        recorder.direct_test(name, bound)
        
        if not tmp_model:
            break
        
        exec(open("predictor.py").read())
        print()


    # train low freq
    name = '../analysis/case6/low-freq.csv'
    recorder.init_test_analysis(name)

    tmp_neg_count_dict = modelizer.make_tmp_data(main_neg_cases_count_dict, low_sort_order)
    tmp_non_count_dict = modelizer.make_tmp_data(main_non_cases_count_dict, low_sort_order)
    tmp_model = copy.deepcopy(main_model)

    while True:
        bound = modelizer.finalize_model(tmp_model, high_freq, low_freq, tmp_neg_count_dict, tmp_non_count_dict, run_case, low_sort_order)
        recorder.direct_test(name, bound)
        
        if not tmp_model:
            break
        
        exec(open("predictor.py").read())
        print()

    # train alpha num
    name = '../analysis/case6/alpha-num.csv'
    recorder.init_test_analysis(name)

    tmp_neg_count_dict = modelizer.make_tmp_data(main_neg_cases_count_dict, default_sort_order)
    tmp_non_count_dict = modelizer.make_tmp_data(main_non_cases_count_dict, default_sort_order)
    tmp_model = copy.deepcopy(main_model)

    bound = modelizer.finalize_model(tmp_model, high_freq, low_freq, tmp_neg_count_dict, tmp_non_count_dict, run_case, default_sort_order)

    power = 1
    alph = 1
    while alph < 100000000000:
        recorder.direct_test(name, alph)
        exec(open("predictor.py").read())
        print()

        alph = 2**power
        power += 1

stop = timeit.default_timer()
print('Time: ', stop - start)
