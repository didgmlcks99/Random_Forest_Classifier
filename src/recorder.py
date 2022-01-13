import csv

def record_text_word_cases(text_word_cases, rec_text_fn):
    with open(rec_text_fn, 'w') as file:
        for cases_text in text_word_cases:
            file.write(str(cases_text))
            file.write('\n')

def record_case_count_dict(count_dict, cound_dict_fn):
    with open(cound_dict_fn, 'w') as file:
        writer = csv.writer(file)

        for case in count_dict:
            row = [case, count_dict[case]]
            writer.writerow(row)

def record_model(model):
    with open('../model/predictor-model.csv', 'w') as file:
        writer = csv.writer(file)

        for case in model:
            row = [case, model[case][0], model[case][1]]
            writer.writerow(row)

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