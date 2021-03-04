# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:43:42 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jdcal import gcal2jd
from collections import Counter
from collections import OrderedDict as OD
from scipy.special import comb


data = pd.read_csv('c:/users/william/desktop/astrostats/SOLAR_FLARES/xray.txt',\
                   header = None, sep = '\\s+', dtype = int)

spath = 'c:/users/william/desktop/astrostats/'                                #set local savepath 


ndata = list(zip(data[0], data[1],data[2], data[3], data[4]))
nbin =  48                                                                    #number of bins for flares in one day histogram
nbin2 = 20                                                                    #number of bins for time between flares histogram
n = len(ndata)

jd_arr = np.nan*np.zeros(n)
for i in range(n):
    year, month, day, hour, minute = ndata[i]
    jd = gcal2jd(year,month,day)[1]                                           #convert to JD
    jd += hour/24 + minute/(60*24)                                            #function doesn't get hours and minutes, add them manually
    jd_arr[i] = jd

jd_arr_full= jd_arr                                                           #keep full JD in array 
tbf = 24*np.diff(jd_arr)                                                      #time between flares in hours

jd_arr = np.floor(jd_arr) - np.floor(jd_arr[0])                               #convert to "days past first day", need integers

#generate dict of number of x-ray flares per day

d = OD(Counter(jd_arr))                                                       #ordered dict nicer to look at 
days_in_d = list(d.keys())                                                    #find which days are in the dictionary
all_days = np.linspace(0,1460,1461)                                           #number of days we SHOULD have in the dictionary
diff= set(all_days) - set(days_in_d)                                          #find out which days are missing
for i in diff:
    d[i] = 0                                                                  #add missing days. Assume zero flares recorded 
    
#we now have a list of each day and how many flares recorded on that day
#Find how many flares ocurred over this time period; get averge number of flares per day

flare_tally = [d[el] for el in d]                                             #number of flares on each day
avg = np.average(flare_tally)                                                 #average number of flares in one day

#now let's count the occurrence of the occurrence rates--how many days have 1 flare, 2 flares, etc.

ocr = OD(Counter(flare_tally))                                                #why OD not order like it should??? Meh...

#want probability distribution of number solar flares per day
#we have 1461 days, so take the elements in ocr and divide by that 

k_arr = [el for el in ocr]
Pk_arr = [ocr[el]/1461 for el in ocr]
n = len(k_arr)

#sort data because otherwise it plots ugly. Wouldn't have to do this if OD did its job...
z = sorted(list(zip(k_arr, Pk_arr)), key = lambda x:x[0])
k_arr_sorted = [i[0] for i in z]
Pk_arr_sorted = [i[1] for i in z]

#compute average probability of flare
p = avg/24

#compute binomial formula and compare to data
N = max(k_arr) # -2 matches what Jim has exactly, but why use -2?

bPk = np.nan*np.zeros(n)
for i in range(n):
    k = k_arr_sorted[i]
    bPk[i] = comb(N,k) * p**k * (1-p)**(N - k)
    
#compute poisson formula and compare to data
pPk = np.nan*np.zeros(n)
t = 24*1
for i in range(n):
    k = k_arr_sorted[i]
    pPk[i] = (p*t)**k * np.exp(-p*t)/ np.math.factorial(k)

#plot raw data because why not
plt.figure(figsize = (12,14))
bins = np.linspace(0,max(jd_arr), nbin + 1)
plt.hist(jd_arr, bins = bins, rwidth = 0.8, color = 'k')
plt.xticks(bins, rotation = 90)
plt.title('Occurrence Rate of X-Ray Flares', size = 24)
plt.xlabel('Days After First Recorded Flare', size = 18)
plt.ylabel('Number of Flares', size = 24)
plt.savefig(spath + 'HW4_4_1_fig.pdf', dpi = 500)
#plt.show()

#plot probability distribution 
plt.figure(figsize = (12,14))
plt.title("Probability of K Flares in One Day", size = 24)
plt.plot(k_arr_sorted, Pk_arr_sorted, color = 'k', marker = 'o', linestyle = '--', label = 'X-Ray Data')
plt.plot(k_arr_sorted, bPk, color = 'red', label = 'Binomial Distribution')
plt.plot(k_arr_sorted, pPk, color = 'blue', label = 'Poisson Distribution')
plt.xlabel('k', size = 18)
plt.ylabel('P(k)', size = 18)
plt.legend()
plt.grid()
plt.savefig(spath + 'HW4_4_2_fig.pdf', dpi = 500)
#plt.show()

#plot time between flares
plt.figure(figsize = (12,14))
plt.title('Time Between Flares', size = 24)
bins2 = np.linspace(0,max(tbf), nbin2+1)
counts, edges, plots = plt.hist(tbf, bins = bins2, color = 'k', rwidth = 0.8)
plt.xticks(bins2, rotation = 90)
plt.xlabel('Time (Hours)', size = 18)
plt.ylabel('Frequency', size = 18)
plt.savefig(spath + 'HW4_4_3_fig.pdf', dpi = 500)
#plt.show()

#plot probability distribution of time between flares
#just take the counts and divide by number of hours in observation period
#After doing all of the below, I realized setting density = True probably would have been easier...oh well...

fx= np.convolve(edges, np.ones(2), 'valid')/2                                 #compute moving average of edges; average 2 successive elements
tbfp = np.array(counts) / (1461*24)                                           #time between flares probability

#define exoponential distribution 
lam = 1/np.average(tbf)
ex = np.linspace(0, max(tbf), 10000)
ey = lam* np.exp(-lam * ex)

plt.figure(figsize = (12,14))
plt.title("Time Between Flares Probability Distribution", size = 24)
plt.bar(fx, tbfp, color = 'k')
plt.plot(ex,ey, color = 'red', label = 'Exponential Distribution')
plt.semilogy()                                                                #optional log(y) axis
plt.xticks(fx, rotation = 90)
plt.xlabel('Average Time (Hours)', size = 18)
plt.ylabel('P(k)', size = 18)
plt.savefig(spath + 'HW4_4_4_fig.pdf', dpi = 500)
#plt.show()

#The time between flares probability distribution histogram seems to roughly follow an exponential distribution
#However, it's not quite steep enough at the smaller times


    

