import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import pickle


def func(x, a, b, c,d, e,f):
    return a + b * x + c / (x) + e * x ** 2+ d / (x**2) + f * x ** 3
def func2(x, a, b, c):
    return a + b * x + c*x**2


def read_txt(filename):
    f = open(filename, "r")
    data = list()
    for line in f.readlines():
        if ";" in line:
            data.append(line.strip().split(";"))
        elif " " in line:
            data.append(line.strip().split(" "))
    array = np.zeros((len(data) - 1, 2))
    for i in range(len(data) - 1):
        if "," in data[i][0]:
            data1temp = data[i][0].split(',')
            data[i][0]= data1temp[0] + "." + data1temp[1]
        if "," in data[i][1]:
            data2temp = data[i][1].split(',')
            data[i][1] = data2temp[0] + "." + data2temp[1]
        array[i][1] = float(data[i][1])
        array[i][0] = float(data[i][0])
    return array


data = read_txt("Data/Elektrolyser.txt")
#data2 = read_txt("Data/pow16.txt")
x_Data = data[:, 0]
y_Data = data[:, 1]
#x_Data2 = data2[:, 0]
#y_Data2 = data2[:, 1]

#with open("Data/demand_data","wb") as fp:
#    pickle.dump(data2,fp)


popt, pcov = curve_fit(func, x_Data, y_Data)
#popt2, pcov2 = curve_fit(func, x_Data2, y_Data2)
lnsp = np.linspace(min(x_Data), max(x_Data), num=50)


test_data1 = func(lnsp, *popt)
#test_data2 = func(lnsp, *popt2)/100
#test_data3 = test_data2*test_data1
#popt3, pcov3 = curve_fit(func2, test_data1, test_data2)

fig = plt.figure()
ax = fig.add_subplot(111)
c = ax.plot(lnsp, test_data1)
#ax = fig.add_subplot(222)
c1 = ax.scatter(x_Data, y_Data)
#ax = fig.add_subplot(223)
#c = ax.scatter(lnsp, test_data3)
#ax = fig.add_subplot(224)
#c = ax.scatter(range(len(x_Data2) ), x_Data2/35040)
#c = ax.plot(x_Data,func(x_Data,*popt),'r-')
#ax1 = fig.add_subplot(122)
#c2 = ax1.scatter(range(len(y_Data2) ), y_Data2)
plt.show()
