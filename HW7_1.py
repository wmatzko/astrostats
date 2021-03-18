# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 04:33:40 2021

@author: William
"""
import numpy as np
#from astropy.stats import jackknife_resampling
#from astropy.stats import jackknife_stats
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(623)

def sigma_CI(s, n, alpha = 0.05, ddof = 1):
    '''
    Function to compute (1-alpha)% confidence interval of sigma via 
        sqrt(n-ddof)/sqrt(b) *s < sigma < sqrt(n-ddof)/sqrt(a)*s
    where a,b is derived from the chi-squared percentile point function (inverse CDF):
        a = chisq_1-alpha/2,n-ddof, b = chisq_alpha/2,n-ddof
    Based on https://online.stat.psu.edu/stat415/book/export/html/810
    '''
    a = stats.chi2.ppf(alpha/2, n-ddof) #slight notational difference in above link and scipy's implementation?
    b = stats.chi2.ppf(1-alpha/2, n-ddof)
    
    lb = np.sqrt(n-ddof)/np.sqrt(b) * s
    ub = np.sqrt(n-ddof)/np.sqrt(a) * s
    
    return (lb,ub)

N=10000
n=9
nbin = 15
mu_true = 130.0
sigma_true = 1.5

x = np.random.normal(mu_true, sigma_true, size=n)
x_avg=np.mean(x)
#print(np.mean(x)) #131.0812930740811

#Apply Case I, we want to first estimate sigma. 
#I thought I was doing Case II with the below method..turns out I wasn't, so I decided to do all 3 cases because why not
print("Case I")
#first use non-parametric bootstrap
draw = np.random.choice(x, replace = True, size = (N,n))

draw_avg = np.mean(draw)
stdev = np.std(draw, ddof = 1, axis = 1)
stdev_avg = np.mean(stdev)
stdev_med = np.median(stdev)

lb,ub = sigma_CI(stdev_med, n)
print("Using the non-parametric bootstrap:\n\
  avg = {0:0.3f}, mean(std) = {1:0.3f}, median(std) = {2:0.3f}\n\
  The 95% CI is ({3:0.3f}, {4:0.3f})\n".format(draw_avg,stdev_avg, stdev_med, lb, ub))

#for fun
#estimate, bias, stderr, conf_interval = jackknife_stats(x, np.var) #not as accurate when doing np.std directly?
#print("Jackknife estimate of sigma is {0:0.3f}\n".format(np.sqrt(estimate))) #seems more accurate than other two methods....at least for this random seed

plt.figure(figsize = (12,14))
plt.title('Non-Parametric Bootstrap for $\\sigma$',size = 24)
bins = np.linspace(np.min(stdev), np.max(stdev), nbin+1)
plt.hist(stdev, bins = bins, color = 'k', rwidth = 0.8)
plt.axvline(stdev_avg, color = 'r', linestyle = '--', label = 'avg = {0:0.3f}'.format(stdev_avg))
plt.axvline(stdev_med, color = 'g', linestyle = '--', label = 'med = {0:0.3f}'.format(stdev_med))
plt.axvline(1.5, color = 'teal', linestyle = '-', label = 'True Value')
plt.axvspan(lb,ub,color = 'gray',alpha = 0.5, label = '95% CI')
plt.xlabel('Sample Standard Deviation, s', size = 18)
plt.ylabel('Frequency', size = 18)
plt.xticks(bins, rotation = 90)
plt.legend()
plt.show()

#Now we try parametric bootstrap
xbar = np.mean(x)
s = np.std(x, ddof = 1)
para_data = []
sample = np.random.normal(xbar,s, size = (N,n))
mu = np.mean(sample)
sigma = np.std(sample, axis = 1)
sigma_avg = np.mean(sigma)
sigma_med = np.median(sigma)

lb,ub = sigma_CI(sigma_med, n)

print("Using the parametric bootstrap:\n\
  avg = {0:0.3f}, mean(std) = {1:0.3f}, med(std) = {2:0.3f}\n\
  The 95% CI is ({3:0.3f}, {4:0.3f})\n".format(mu, sigma_avg, sigma_med,lb,ub))

plt.figure(figsize = (12,14))
plt.title('Parametric Bootstrap for $\\sigma$', size=24)
bins = np.linspace(np.min(sigma), np.max(sigma), nbin+1)
plt.hist(sigma, bins = bins, color = 'k', rwidth = 0.8)
plt.axvline(sigma_avg, color = 'red', linestyle = '--', label = 'avg = {0:0.3f}'.format(sigma_avg))
plt.axvline(sigma_med, color = 'g', linestyle = '--', label = 'med = {0:0.3f}'.format(sigma_med))
plt.axvline(1.5, color = 'teal', linestyle = '-', label = 'True Value')
plt.axvspan(lb,ub,color = 'gray',alpha = 0.5, label = '95% CI')
plt.xticks(bins, rotation = 90)
plt.xlabel('Sample Standard Deviation, s', size = 18)
plt.ylabel('Frequency', size = 18)
plt.legend()
plt.show()

#stdev_med is the closest to the true value out of all of the measures, so we'll use that
sigma = stdev_med
#print(stdev_med, 'sigma')

#compute test statistic value z
z = (x_avg - mu_true)/(sigma/np.sqrt(n))

#compute critical value of z at (1-alpha)% confidence to determine rejection region
alpha = 0.01
z_crit_low = stats.norm.ppf(alpha/2)
z_crit_upp = stats.norm.ppf(1-alpha/2)

reject_null = [ z<= z_crit_low or z>=z_crit_upp][0]
print("The z value is {0:0.3f}. The z-score range with alpha = {1:0.3f} is ({2:0.3f}, {3:0.3f}).\n\
  Decision to reject null hypothesis: {4}\n".format(z, alpha, z_crit_low,z_crit_upp, reject_null))

#We now do Case II
print("Case II")
x_std = np.std(x, ddof = 1)
z = (x_avg - mu_true)/(x_std/np.sqrt(n))

#Use same critical values as above
reject_null = [ z<= z_crit_low or z>=z_crit_upp][0]
print("The z value is {0:0.3f}. The z-score range with alpha = {1:0.3f} is ({2:0.3f}, {3:0.3f}).\n\
  Decision to reject null hypothesis: {4}\n".format(z, alpha, z_crit_low,z_crit_upp, reject_null))

#Now do Case III

print("Case III")

#t value is identical to z value here, only rejection region changes 
t = z

t_crit_low = stats.t.ppf(alpha/2,n-1)
t_crit_upp = stats.t.ppf(1-alpha/2,n-1)

reject_null = [ t<= t_crit_low or t>=t_crit_upp][0]

print("The t value is {0:0.3f}. The t-score range with alpha = {1:0.3f} is ({2:0.3f}, {3:0.3f}).\n\
  Decision to reject null hypothesis: {4}\n".format(z, alpha, t_crit_low,t_crit_upp, reject_null))