# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 03:01:49 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

#set problem parameters
mu = 0
sigma = 1
N = 10000                                                                     #number of experiments
n = 10                                                                        #number of draws in each experiment
nbins = 50                                                                    #number of bins for histogram

sample= np.random.normal(mu,sigma,(N,n))                                      #generate Gaussian dist size Nxn
avg = np.average(sample, axis = 1)                                            #compute average for each of the n trials
sb2_arr = np.nan*np.zeros(N)                                                  #initialize array to hold sb2 values

for j in range(N):
    sb2 = 1/(n)*np.sum([(sample[j][i] - avg[j])**2 for i in range(n)])        #compute sb2
    sb2_arr[j] = sb2

sb2_avg = np.average(sb2_arr)                                                 #compute average and variance
sb2_var = np.var(sb2_arr)

#make plot
plt.figure(figsize = (12,14))
# 9 decimal points does not make sense. If you run this 2x, only the
# value up to about two decimal points stays the same.
plt.title('Biased Sample Standard Deviation, $\\overline{{S_b^2}} = {0:0.3f}$, Var($S_b^2$) = {1:0.9f}'.format(sb2_avg, sb2_var), size = 20)
bins = np.linspace(np.min(sb2_arr), np.max(sb2_arr), nbins+1)
plt.hist(sb2_arr, bins, color = 'k', rwidth = 1)
plt.xlabel('$S_b^2$', size = 18)
plt.ylabel('Frequency', size = 18)
#plt.xticks(bins, rotation = 90)
#plt.grid()
plt.show()
