#!/usr/bin/python

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sp


# a = 1.05631874, b = 0.3153276

def func(x, a, b):
    return a * np.tanh(b * x)


if __name__ == "__main__":

    try:
        fp = open("data.txt")
        line = fp.readline()
        xdata = []
        ydata = []
        vec = line.strip().split('\t')
        xdata.append(float(vec[0]))
        ydata.append(float(vec[-1]))
        while line:            
            line = fp.readline()
            if line == '': break
            
            vec = line.strip().split('\t')
            
            xdata.append(float(vec[0]))
            ydata.append(float(vec[-1]))

    finally:
        fp.close()

    p0 = sp.array([0.1, 0.1])
    
    coeffs, cov = curve_fit(func, np.array(xdata), np.array(ydata), p0)

    
    plt.figure(figsize=(6,4))
    plt.scatter(xdata, ydata, label='Data')
    plt.plot(xdata, func(np.array(xdata), np.array(coeffs[0]),
                         coeffs[1]), label='Fitted function')
    plt.legend(loc='best')
    plt.show()

    print(coeffs)
    print(cov)

    for x in xdata:
        print(x, func(np.array(x), coeffs[0], coeffs[1]))

