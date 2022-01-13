import csv
import recorder

def count_text_word_cases(text_word_cases, word_cases_count_dict):
    for case in text_word_cases:
        if case in word_cases_count_dict.keys():
            word_cases_count_dict[case] += 1
        else:
            word_cases_count_dict[case] = 1

def make_tmp_data(cases_count_dict, sort_order):
    tmp_count_dict = cases_count_dict.copy()

    tmp = sort_word_cases(tmp_count_dict, sort_order)
    tmp_count_dict.clear()
    tmp_count_dict = tmp

    return tmp_count_dict

def sort_word_cases(tmp_count_dict, sort_order):
    res = {key: val for key, val in sorted(tmp_count_dict.items(), key = lambda ele: ele[1], reverse = sort_order)}
    return res

def mk_model(alpha_num, neg_texts, non_texts, neg_cases_count_dict, non_cases_count_dict):
    model = {}
    
    init_model(model, neg_cases_count_dict)
    init_model(model, non_cases_count_dict)

    work_model(alpha_num, model, neg_texts, non_texts)

    return model

def init_model(model, count_dict):
    for case in count_dict:
        if not case in model.keys():
            model[case] = [0.0, 0.0]

def work_model(alpha_num, model, neg_texts, non_texts):
    print("working on prediction model...")
    i = 1
    model_size = len(model)
    for case in model:
        neg_count = 0
        non_count = 0
        for text in neg_texts:
            if case in text:
                neg_count += 1
        
        for text in non_texts:
            if case in text:
                non_count += 1
    
        model[case][0] = (neg_count+alpha_num) / (len(neg_texts)+alpha_num)
        model[case][1] = (non_count+alpha_num) / (len(non_texts)+alpha_num)

        print('> ' + str(i) + '/' + str(model_size) +  ' ' + "{:.2f}".format((i/model_size)*100) + '%: ' + case + " [" + str(model[case][0]) + ", " + str(model[case][1]) + "]")
        i += 1
        
    print("prediction model done...")

def finalize_model(model, high_freq, low_freq, neg_cases_count_dict, non_cases_count_dict, run_case, order):
    if run_case == True:
        stopword_rm(model, high_freq, low_freq, neg_cases_count_dict, non_cases_count_dict)
    else:
        bound = stopword_lim(model, neg_cases_count_dict, non_cases_count_dict, order)
        return bound
    
    recorder.record_case_count_dict(neg_cases_count_dict, '../model/train.negative.count.csv')
    recorder.record_case_count_dict(non_cases_count_dict, '../model/train.non-negative.count.csv')
    recorder.record_model(model)

def stopword_rm(model, high_freq, low_freq, neg_dict, non_dict):
    neg_case_list = list(neg_dict.keys())
    for case in neg_case_list:
        if neg_dict[case] >= high_freq:
            neg_dict.pop(case)
            
            if non_dict:
                if case in non_dict.keys():
                    non_dict.pop(case)
            
            if model:
                if case in model.keys():
                    model.pop(case)

        elif neg_dict[case] <= low_freq:
            neg_dict.pop(case)

            if non_dict:
                if case in non_dict.keys():
                    non_dict.pop(case)
            
            if model:
                if case in model.keys():
                    model.pop(case)
    
    non_case_list = list(non_dict.keys())
    for case in non_case_list:
        if non_dict[case] >= high_freq:
            non_dict.pop(case)
            
            if neg_dict:
                if case in neg_dict.keys():
                    neg_dict.pop(case)
            
            if model:
                if case in model.keys():
                    model.pop(case)
                
        elif non_dict[case] <= low_freq:
            non_dict.pop(case)

            if neg_dict:
                if case in neg_dict.keys():
                    neg_dict.pop(case)
            
            if model:
                if case in model.keys():
                    model.pop(case)

def stopword_lim(model, neg_dict, non_dict, order):
    neg_keys = list(neg_dict.keys())
    non_keys = list(non_dict.keys())

    neg_bound = 0
    non_bound = 0
    if not neg_dict:
        neg_bound = -1
    else:
        neg_bound = neg_dict[neg_keys[0]]
    
    if not non_dict:
        non_bound = -1
    else:
        non_bound = non_dict[non_keys[0]]
    
    bound = get_bound(neg_bound, non_bound, order)
    stopword_lim_rm(model, neg_dict, non_dict, bound)

    return bound

def get_bound(neg_bound, non_bound, order):
    bound = 0

    if neg_bound == -1 and non_bound == -1:
        bound = -1
    elif neg_bound == -1:
        bound = non_bound
    elif non_bound == -1:
        bound = neg_bound
    elif order == True:
        if neg_bound >= non_bound:
            bound = neg_bound
        else:
            bound = non_bound
    elif order == False:
        if neg_bound <= non_bound:
            bound = neg_bound
        else:
            bound = non_bound
    
    return bound

def stopword_lim_rm(model, neg_dict, non_dict, bound):
    neg_case_list = list(neg_dict.keys())
    for case in neg_case_list:
        if neg_dict[case] == bound:
            neg_dict.pop(case)
            
            if non_dict:
                if case in non_dict.keys():
                    non_dict.pop(case)
            
            if model:
                if case in model.keys():
                    model.pop(case)

        else:
            break
    
    non_case_list = list(non_dict.keys())
    for case in non_case_list:
        if non_dict[case] == bound:
            non_dict.pop(case)
            
            if neg_dict:
                if case in neg_dict.keys():
                    neg_dict.pop(case)
            
            if model:
                if case in model.keys():
                    model.pop(case)
                
        else:
            break

def print_result_info(result):
    
    tp = result[0]
    tn = result[1]
    fp = result[2]
    fn = result[3]

    acc = result[4]
    prec = result[5]
    rec = result[6]
    
    with open('../analysis/direction.csv', 'r') as file:
        csvFile = csv.reader(file)

        for row in csvFile:
            if len(row) == 0:
                continue

            with open(row[0], 'a') as analysis_file:
                writer = csv.writer(analysis_file)

                line = [row[1], tp, fn, fp, tn, acc, prec, rec]
                print(row[0] + ' : ', end='')
                print(line)
                writer.writerow(line)