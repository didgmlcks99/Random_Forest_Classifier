import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(name):
    print(name)

    with open(name, mode = 'r') as file:
        x_axis = []
        tp = []
        fn = []
        fp = []
        tn = []

        acc = []
        prec = []
        rec = []

        csvFile = csv.reader(file)

        for lines in csvFile:
            if lines[0][0].isalpha():
                continue
            
            # if 100 < int(lines[0]) < 170 :
            x_axis.append(float(lines[0]))
            tp.append(float(lines[1]))
            fn.append(float(lines[2]))
            fp.append(float(lines[3]))
            tn.append(float(lines[4]))

            acc.append(float(lines[5])*100)
            prec.append(float(lines[6])*100)
            rec.append(float(lines[7])*100)
        
        plt.plot(x_axis, tp, label='tp')
        # plt.plot(x_axis, fn, label='fn')
        # plt.plot(x_axis, fp, label='fp')
        plt.plot(x_axis, tn, label='tn')

        plt.plot(x_axis, acc, label='acc')
        plt.plot(x_axis, prec, label='prec')
        plt.plot(x_axis, rec, label='rec')

        folders = name.split('/')
        file = folders[3].split('.')
        title = file[0]

        plt.title(title + " bound")
        plt.xlabel('value')
        plt.ylabel('percent')
        plt.legend()
        plt.show()

plot_graph('../analysis/case1/high-freq.csv')
plot_graph('../analysis/case1/low-freq.csv')
# plot_graph('../analysis/case1/alpha-num.csv')