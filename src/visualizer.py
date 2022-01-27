import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(name):
    print(name)

    with open(name, mode = 'r') as file:
        stat = {}
        res = {}

        x_num = []

        tp = []
        fn = []
        fp = []
        tn = []

        accuracy = []
        precision = []
        recall = []

        csvFile = csv.reader(file)

        for lines in csvFile:
            if lines[0][0].isalpha():
                stat[lines[1]] = 0
                stat[lines[2]] = 0
                stat[lines[3]] = 0
                stat[lines[4]] = 0

                res[lines[5]] = 0
                res[lines[6]] = 0
                res[lines[7]] = 0

                continue

            
            x_num.append(int(lines[0].split('/')[1]))
            tp.append(float(lines[1]))
            fn.append(float(lines[2]))
            fp.append(float(lines[3]))
            tn.append(float(lines[4]))

            accuracy.append(float(lines[5])*100)
            precision.append(float(lines[6])*100)
            recall.append(float(lines[7])*100)
        
        
        stat['tp'] = get_average(tp)
        stat['fn'] = get_average(fn)
        stat['fp'] = get_average(fp)
        stat['tn'] = get_average(tn)

        res['accuracy'] = get_average(accuracy)
        res['precision'] = get_average(precision)
        res['recall'] = get_average(recall)

        # folders = name.split('/')
        # file = folders[3].split('.')
        # title = file[0]

        # x = np.arange(4)
        # stats = list(stat.keys())
        # vals = list(stat.values())

        # plt.xlabel('stat')
        # plt.ylabel('avg')
        # plt.title(title + "stats")

        # plt.bar(x, vals)
        # plt.xticks(x, stats)

        # plt.show()

        
        
        # x = np.arange(3)
        # stats = list(res.keys())
        # vals = list(res.values())

        # plt.xlabel('results')
        # plt.ylabel('avg')
        # plt.title(title + 'results')

        # plt.bar(x, vals)
        # plt.xticks(x, stats)

        # plt.show()

        # x = ['c1', 'c2', 'c3', 'c4']
        x = x_num
        acc_y = accuracy
        prec_y = precision
        rec_y = recall

        plt.plot(x, acc_y)
        plt.plot(x, prec_y)
        plt.plot(x, rec_y)

        plt.legend(['accuracy', 'precision', 'recall'])

        plt.show()

def get_average(list):
    size = len(list)
    tot_sum = 0

    for n in list:
        tot_sum += n
    
    return n/size

plot_graph('../analysis/main/main.csv')
# plot_graph('../analysis/case5/low-freq.csv')
# plot_graph('../analysis/case5/alpha-num.csv')