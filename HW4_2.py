# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:23:36 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from collections import Counter
np.random.seed(42)

N_exp = 10000                                            #number of experiments
N = 100                                                  #number of trials in each experiment
k_arr = np.linspace(0,N,N+1)                             #number of successes
n = len(k_arr)                                           
p = 0.4                                                  #success rate

Pk_true_arr = np.nan*np.zeros(n)                         #Store Pk
Pk_exp_arr = np.nan*np.zeros(n)
exp_success_arr = np.nan*np.zeros(N_exp)                 #Store binomial experiment results

exp_success_arr2 = np.nan*np.zeros((N_exp, n))           #Store second 'binomial' experiment results
Pk_exp_arr2 = np.nan*np.zeros(n)

for i in range(n):                              #compute Pk with the formula for given k
    Pk_true = comb(N,k_arr[i]) * p**k_arr[i] * (1-p)**(N-k_arr[i])
    Pk_true_arr[i] = Pk_true
    
for i in range(N_exp):                                   #Simulate 10,000 experiments
    pk_i = np.random.random(N)                           #generate random number between 0-1
    mask = np.where(pk_i <= p)                           #find which elements are successes
    N_success = len(pk_i[mask])                          #number of total successes
    exp_success_arr[i] = N_success


        
for i in range(N_exp):       
    p = 0.4                            #Trials for variable p
    for j in range(n):                                   #Trials done one at a time, don't know bette way
        pk_j = np.random.random(1)
        if j > 1:
            #print(exp_success_arr2[i])
            if (exp_success_arr2[i][j-1] ==1) and (exp_success_arr2[i][j-2] == 1):
                #print('bang')
                p += 0.1*p
            else:
                #print('flop')
                p = 0.4

        mask = np.where(pk_j <= p)
        N_success = len(pk_j[mask])
        exp_success_arr2[i][j] = N_success
        
exp_count = Counter(exp_success_arr)                     #Count number of successes 
exp_count2 = Counter(np.sum(exp_success_arr2, axis = 1)) 

for i in range(len(k_arr)):                              #loop through counter dict
    if k_arr[i] in exp_count.keys():
        Pk_exp_arr[i] = exp_count[k_arr[i]]/N_exp        #if k is in dict, store it in Pk_exp_arr
    else:
        Pk_exp_arr[i] = 0                                #else, store 0
    if k_arr[i] in exp_count2.keys():                    #likewise for other array
        Pk_exp_arr2[i] = exp_count2[k_arr[i]]/N_exp
    else:
        Pk_exp_arr2[i] = 0
    
    
plt.figure(figsize = (12,14))                            #plot results
plt.title('Bernoulli Trials', size = 24)
plt.plot(k_arr, Pk_true_arr, marker = 'o', linestyle = '--', color = 'r', label = 'Formula')
plt.plot(k_arr, Pk_exp_arr, marker = 'o', linestyle = '--', color = 'b', label = 'RNG')
plt.plot(k_arr, Pk_exp_arr2, marker = 'o', linestyle = '--', color = 'g', label = 'Variable P')
plt.xlabel('k', size = 18)
plt.ylabel('P(k)', size = 18)
plt.grid()
plt.legend()
#plt.show()
plt.savefig('c:\\users\\william\\desktop\\astrostats\\' + 'HW4_2.pdf', dpi = 500)

