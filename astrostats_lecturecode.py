# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:48:05 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

n= 2
N = 10000
x = np.random.randn(n, N)
xbar = np.mean(x, axis = 0)
print(x, x.shape, xbar[0])

x2= np.random.randn(N,n)
xbar2 = np.mean(x2, axis = 1)
print(x2, x2.shape, xbar2[0])

plt.figure()
plt.hist(xbar)
plt.show()

plt.figure()
plt.hist(xbar2)
plt.show()
