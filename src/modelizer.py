import csv

def print_train_info(neg_text, non_text, high_freq, low_freq, alpha_num, neg_word_count, non_word_count, predictor_model):

    lines = []
    
    lines.append("*************** trainer info ***************")

    lines.append(("stopword frequency upper bound: " + str(high_freq)))
    lines.append(("stopword frequency lower bound: " + str(low_freq)))
    print("stopword frequency upper bound: " + str(high_freq))
    print("stopword frequency lower bound: " + str(low_freq))

    lines.append(("alpha number: " + str(alpha_num)))
    print("alpha number: " + str(alpha_num))

    lines.append(("neg text size: " + str(len(neg_text))))
    lines.append(("number of words from neg (after normalization and stemming): " + str(len(neg_word_count))))
    print("neg text size: " + str(len(neg_text)))
    print("number of words from neg (after normalization and stemming): " + str(len(neg_word_count)))
    
    lines.append(("non text size: " + str(len(non_text))))
    lines.append(("number of words from non (after normalization and stemming): " + str(len(non_word_count))))
    print("non text size: " + str(len(non_text)))
    print("number of words from non (after normalization and stemming): " + str(len(non_word_count)))

    lines.append(("total number of words in predictor model: " + str(len(predictor_model))))
    print("total number of words in predictor model: " + str(len(predictor_model)))

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
    
    lines = []

    lines.append("\n*************** tester info ***************")

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

    with open('../model/output-info.txt', 'a') as file:
        file.write('\n'.join(lines))
        file.write('\n\n\n')