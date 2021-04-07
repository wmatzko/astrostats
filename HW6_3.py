# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 02:22:02 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
np.random.seed(42)

#Assume data roughly normal so that we can model it with t-distribution
def t_interval(xbar, s):
    '''
    t-distribution 95% confidence interval for xbar
    '''
    ns = n#len(xbar)
    lbound = xbar-2.262 * s/np.sqrt(ns)
    ubound = xbar+2.262 * s/ np.sqrt(ns)
    return (lbound, ubound)

data = [-0.54603241,-0.40652794,-0.11570264,-1.26244673,-1.38616981,\
        -0.44812319,0.82880132,0.79937713,-1.098357,0.38530288]

n = len(data)
N = 10000                                                                    #Number of draws
nbin = 30

plt.figure(figsize = (12,14))

plt.hist(data, bins=10)
plt.show()

data_mean = np.mean(data)
print(data_mean)

draws = np.random.choice(data, replace=True, size = (N,n))                  #draw 10 samples with replacement 10,000 times

xbars = np.average(draws, axis = 1)                                         #compute sample averages
s = np.std(draws, axis = 1, ddof = 1)                                       #compute sample standard deviations
draw_mean = np.mean(draws)                                                  #Compute mean of entire draw data

#Apply first method to find 95% CI
delta = xbars - data_mean                                                   #Compute sample means my data means
percentile = np.percentile(delta, [2.5, 97.5])                              #Find percentiles
CI = (data_mean-percentile[1], data_mean - percentile[0])                   #Calculate CI
print("The CI from the first method is ({0:0.3f}, {1:0.3f})\n".format(CI[0], CI[1]))

# This is the method I was looking for. What you found is that
# other methods give wider CIs. This is due to the heavy tails that you found
# for the upper/lower bounds. It is great that you checked multiple methods.
# Note that for larger n, I expect the methods to give similar results.

#Apply second method using t-distribution
t_arr = []                                                                  #hold 95% CIs

for i in range(len(xbars)):                                                 #Compute CI for each xbar
    xb_el = xbars[i]
    s_el = s[i]
    el = t_interval(xb_el, s_el)
    t_arr.append(el)

lb_dist = [el[0] for el in t_arr]                                           #Make list of lower/upper bounds
ub_dist = [el[1] for el in t_arr]

lb_avg, lb_med = np.average(lb_dist), np.median(lb_dist)                    #Find median and average lower/upper bounds
ub_avg,ub_med = np.average(ub_dist), np.median(ub_dist)

print("The average 95% CI using the second method is ({0:0.3f}, {1:0.3f})".format(lb_avg,ub_avg))
print("The median 95% CI using the second method is ({0:0.3f}, {1:0.3f})\n".format(lb_med, ub_med))


#Apply third method; just use SciPy
sci_ci = st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))
print("The 95% CI using SciPy is ({0:0.3f}, {1:0.3f})".format(sci_ci[0], sci_ci[1]))

lerr1,lmean,lerr2 = np.percentile(lb_dist, [16,50,84])                      #68% coverage of data; error estimate for upper/lower bounds
uerr1,umean,uerr2 = np.percentile(ub_dist, [16,50,84])

#make histogram of bootstrap data
plt.figure(figsize = (12,14))
plt.title('Bootstrap Data Distribution, Average 95% CI $({0:0.3f} \leq \mu \leq {1:0.3f})$'.format(lb_avg,ub_avg), size = 20)
xbarbin = np.linspace(np.min(xbars), np.max(xbars), nbin+1)
plt.hist(xbars, color = 'k', bins = xbarbin, density = True)
plt.xlabel('$\\overline{{X}}$', size = 18)
plt.ylabel('Probability Density',size = 18)

#CI from first method
plt.axvline(CI[0], color = 'purple', linestyle = '--', label = 'First Method')
plt.axvline(CI[1], color = 'purple', linestyle = '--')

#CIs for second method
plt.axvline(lb_avg, color = 'red', linestyle = '--', label = 'Mean 95% CI')
plt.axvline(ub_avg, color = 'red', linestyle = '--')
plt.axvline(lb_med, color = 'blue', linestyle = '--', label = 'Median 95% CI')
plt.axvline(ub_med, color = 'blue', linestyle = '--')
plt.axvline(draw_mean, color = 'green', linestyle = '-', label = 'Average = {0:0.3f}'.format(draw_mean))

#CI from SciPy
plt.axvline(sci_ci[0], color = 'sandybrown', label = 'SciPy Method', linestyle = '--')
plt.axvline(sci_ci[1], color = 'sandybrown', linestyle = '--')
plt.axvspan(lerr1,lerr2, color = 'gray', alpha = 0.5, label = '68% Bound CI')
plt.axvspan(uerr1,uerr2, color = 'gray', alpha = 0.5)
plt.legend()
plt.show()

#Make histogram of lower and upper bound distributions

plt.figure(figsize = (12,14))
plt.title('Bounds Distribution of 95% Confidence Interval', size = 24)
combined = np.ndarray.flatten(np.array(t_arr))
bins = np.linspace(np.min(combined), np.max(combined), nbin +1)

plt.hist(lb_dist, color = 'red', label = 'Lower Bound: ${0:0.3f}^{{{1:0.3f}}}_{{{2:0.3f}}}$'.format(lmean, lerr2-lmean, lmean-lerr1),\
         bins = bins, alpha = 0.5, density=True)
plt.hist(ub_dist, color = 'blue', label = 'Upper Bound: ${0:0.3f}^{{{1:0.3f}}}_{{{2:0.3f}}}$'.format(umean, uerr2-umean, umean-uerr1),\
         bins = bins, alpha = 0.5, density=True)

plt.xlabel('Bound Value', size = 18)
plt.ylabel('Probability Density', size = 18)

#Draw lines encompassing 68% of the data
plt.axvline(lerr1, color = 'darkred', linestyle = '--')
plt.axvline(lmean, color = 'darkred', linestyle = '-')
plt.axvline(lerr2, color = 'darkred', linestyle = '--')

plt.axvline(uerr1, color = 'darkblue', linestyle = '--')
plt.axvline(umean, color = 'darkblue', linestyle = '-')
plt.axvline(uerr2, color = 'darkblue', linestyle = '--')

plt.legend()
plt.show()







