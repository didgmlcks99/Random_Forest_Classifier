import csv

def print_train_info(neg_text, non_text, high_freq, low_freq, alpha_num, neg_word_count, non_word_count, predictor_model):

    lines = []
    
    lines.append("*************** trainer info ***************")

    lines.append(("stopword frequency upper bound: " + str(high_freq)))
    lines.append(("stopword frequency lower bound: " + str(low_freq)))

    lines.append(("alpha number: " + str(alpha_num)))

    lines.append(("neg text size: " + str(len(neg_text))))
    lines.append(("number of words from neg (after normalization and stemming): " + str(len(neg_word_count))))
    
    lines.append(("non text size: " + str(len(non_text))))
    lines.append(("number of words from non (after normalization and stemming): " + str(len(non_word_count))))

    lines.append(("total number of words in predictor model: " + str(len(predictor_model))))

    with open('../model/output-info.txt', 'a') as file:
        file.write('\n'.join(lines))

def model_csv(model):
    with open('../model/predictor-model.csv', 'w') as file:
        writer = csv.writer(file)

        for word in model:
            line = [word, model[word][0], model[word][1]]
            writer.writerow(line)

def texts_data(texts, texts_name):
    with open(texts_name, 'w') as file:
        for text in texts:
            file.write(' '.join(text))
            file.write('\n')

def count_data(word_count, count_name):
    with open(count_name, 'w') as file:
        writer = csv.writer(file)

        for word in word_count:
            line = [word, word_count[word]]
            writer.writerow(line)

def print_result_info(neg_text, non_text, model, result):
    
    tp = result[0]
    tn = result[1]
    fp = result[2]
    fn = result[3]

    acc = result[4]
    prec = result[5]
    rec = result[6]

    lines = []

    lines.append("\n*************** tester info ***************")

    lines.append(("number of word in model: " + str(len(model))))
    lines.append(("number of neg text: " + str(len(neg_text))))
    lines.append(("number of non text: " + str(len(non_text))))

    lines.append("tp,fn,fp,tn,accuracy,precision,recall: ")
    lines.append(str(tp) + ", " + str(fn) + ", " + str(fp) + ", " + str(tn) + ", " + str(acc) + ", " + str(prec) + ", " + str(rec))

    with open('../model/output-info.txt', 'a') as file:
        file.write('\n'.join(lines))
        file.write('\n\n\n')
    
    with open('../analysis/direction.csv', 'r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            if len(lines) == 0:
                continue

            with open(lines[0], 'a') as analysis_file:
                writer = csv.writer(analysis_file)

                line = [lines[1], tp, fn, fp, tn, acc, prec, rec]
                print(lines[0] + ' : ', end='')
                print(line)
                writer.writerow(line)


