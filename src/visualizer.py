import csv
import matplotlib.pyplot as plt
import numpy as np

def get_points(name):
    print(name)

    with open(name, mode = 'r') as file:
        x_axis = []
        neg_y = []
        non_y = []

        csvFile = csv.reader(file)

        for lines in csvFile:
            if lines[0][0].isalpha():
                continue
            
            if 100 < int(lines[0]) < 170 :
                x_axis.append(float(lines[0]))
                neg_y.append(float(lines[1]))
                non_y.append(float(lines[2]))
        
        return [x_axis, neg_y, x_axis, non_y]

high_point = get_points('../analysis/lowercase/high-freq.csv')
high_neg_x = np.array(high_point[0])
high_non_x = np.array(high_point[2])
high_neg_y = np.array(high_point[1])
high_non_y = np.array(high_point[3])

low_point = get_points('../analysis/lowercase/low-freq.csv')
low_neg_x = np.array(low_point[0])
low_non_x = np.array(low_point[2])
low_neg_y = np.array(low_point[1])
low_non_y = np.array(low_point[3])

alpha_point = get_points('../analysis/lowercase/alpha-num.csv')
alpha_neg_x = np.array(alpha_point[0])
alpha_non_x = np.array(alpha_point[2])
alpha_neg_y = np.array(alpha_point[1])
alpha_non_y = np.array(alpha_point[3])

# alpha_neg_x, alpha_neg_y, alpha_non_x, alpha_non_y

plt.plot(high_neg_x, high_neg_y, label='high neg')
plt.plot(high_non_x, high_non_y, label='high non')
plt.plot(low_neg_x, low_neg_y, label='low neg')
plt.plot(low_non_x, low_non_y, label='low non')
plt.xlabel('value')
plt.ylabel('percent')
plt.legend()
plt.show()