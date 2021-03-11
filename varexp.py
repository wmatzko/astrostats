# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:48:32 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt

N = 10000
n = 10
nbin = 30

x = np.random.randn(N,n)

avg = np.average(x, axis = 1)
var = np.var(x, axis = 1, ddof = 1)


plt.figure(figsize = (12,14))
plt.title('Variance Sampling Test')
bins = np.linspace(min(var), max(var), nbin)
plt.hist(var, bins = bins, color = 'k')
plt.xlabel('Var(x)')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize = (12,14))
plt.title('Average Sampling Test')
bins = np.linspace(min(avg), max(avg), nbin)
plt.hist(avg, bins = bins, color = 'k')
plt.xlabel('Avg(x)')
plt.ylabel('Frequency')
plt.show()

x = np.random.binomial(n = 1, p = 0.5, size = (2, 1000))
print(x)