# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 04:56:27 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
np.random.seed(42)

#please see HW4_3.pdf for explanation of the logic behind this code.

def cz(x):
    '''
    Count number of zeros between nonzero elements
    This returns the number of minutes between flares if the correct array is passed
    '''
    count = 0
    c_arr = []
    for i in range(len(x)):
        el = x[i]
        if el == 0:
            count += 1
        else:
            if count>0:
                c_arr.append(count)
            count = 0
    return c_arr

mu = 900 / (1000 * 24)                                       #average probability of flare occurring in one hour
N_exp = 1000                                                 #number of experiments
k_max = 10
k_arr = np.linspace(0,k_max, k_max + 1)                      #arbitrary number of successes to plot
n = len(k_arr)
n_bin = 20                                                   #number of bins in histogram

#can expect 1000 * 24 / 900 ~27 hours for 1 flare to show
#so, make dt = 1 hour and N = 24
N = 24
dt = 1
t = N*dt

Pk_arr = np.nan*np.zeros(n)                                  #initialize arrays to hold plotting info
Pk_exp_arr = np.nan*np.zeros(n)

                                                             #Compute with formula
for i in range(n):
    k = k_arr[i]
    Pk = (mu * t)**k * np.exp(-mu*t) / np.math.factorial(k)
    Pk_arr[i] = Pk

                                                             #Compute with RNG to minute resolution
cut = np.exp(-mu/60)                                         #Poisson threshold in minutes
                                                             #If do hours, with this mu can get multiple flares in one hour

t_arr = np.zeros((N_exp,24,60))
for l in range(t_arr.shape[0]):
    for i in range(t_arr.shape[1]):
        for j in range(t_arr.shape[2]):
            r = np.random.random()
            if r >= cut:
                k = 1                                        #Assume resolution small enough to do Bernoulli trials
            else:
                k = 0
            t_arr[l][i][j] = k

#print(t_arr)
hour_tally = np.sum(t_arr, axis = 2)                         #tally number of flares ocurring each hour
day_tally = np.sum(hour_tally, axis = 1)                     #tally number of flares occurring each day
total_tally = np.sum(day_tally)                              #total number of flares
d=Counter(day_tally)                                         #Count how many days have n flares

#Extract data from Counter and populate probability array

for i in range(n):
    el = k_arr[i]
    if el in d:
        Pk_exp_arr[i] = d[el]/N_exp
    else:
        Pk_exp_arr[i] = 0
#look at entire chain, don't break into day blocks
tbf = []                                                     #hold time between flares for all days and all experiments
for i in range(1):
    #day_block = t_arr[i]
    #day_chain = np.ndarray.flatten(np.array(day_block))     #flatten each hour long sub-block into one long block

    chain = np.ndarray.flatten(np.array(t_arr))
    times = cz(chain)
    if len(times) > 0:
        for j in times:                                      #Add to list on element-by-element basis
            tbf.append(j/60) #convert to hours

#make figures
spath = 'c:/users/william/desktop/astrostats/'              #set local savepath
plt.figure(figsize = (12,14))
plt.title('Number of Flares in a Day', size = 24)
plt.plot(k_arr, Pk_arr, color = 'r', marker = 'o', linestyle = '--', label = 'Poisson Formula')
plt.plot(k_arr, Pk_exp_arr, color = 'b', marker = 'o', linestyle = '--', label = 'RNG')
plt.xlabel('k', size = 18)
plt.ylabel('P(k)', size = 18)
plt.legend()
plt.grid()
#plt.savefig(spath + 'HW4_3_1_fig.pdf', dpi = 500)
#plt.show()

plt.figure(figsize = (12,14))
plt.title('Time Between Flares',size = 24)
bins = np.linspace(0, max(tbf), n_bin + 1)
plt.hist(tbf, bins = bins, color = 'k', rwidth = 0.8)
plt.xlabel('Time (Hours)', size = 18)
plt.ylabel('Frequency', size = 18)
plt.xticks(bins, rotation = 90)
#plt.savefig(spath + 'HW4_3_2_fig.pdf', dpi = 500)
#plt.show()


#equivalent method using numpy poisson generator to my generator
#pt =np.random.poisson(900 / (1000 * 24 * 60), (1000, 24,60))
#hour_tally = np.sum(pt, axis = 2) #tally the number of flares ocurring each hour
#day_tally = np.sum(hour_tally, axis = 1) #tally number of flares ocurring each day
#total_tally = np.sum(day_tally) #tally total number of flares
#print(hour_tally)
#print(day_tally)
#print(total_tally, np.average(day_tally))