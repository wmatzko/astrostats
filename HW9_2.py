# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:20:03 2021

@author: William
"""
###Raw code from Github

import numpy as np   #import necessary packages 
import scipy
import scipy.stats # I needed to add this line so the scipy stats call would work...

x = np.array([-1.22, -1.17, 0.93, -0.58, -1.14]) #define arbitrary x and y data; taken from Walls & Jenkins pg 81
y = np.array([1.03, -1.59, -0.41, 0.71, 2.10])

print(np.mean(y)-np.mean(x)) #compute the difference of the means for each data

sigma_x = np.std(x,ddof=1) #compute the standrd deviation of xdata, ddof = 1 reminds us this is a sample from a population
sigma_y = np.std(y,ddof=1) #likewise for y data
sigma_xy_pooled = np.sqrt(sigma_x**2 + sigma_y**2)/np.sqrt(2) #weighted stnd dev; larger sigma has more weight. Just a definition
print(sigma_xy_pooled)

t = (np.mean(y)-np.mean(x))/(sigma_xy_pooled*np.sqrt(2/5)) #compute t-statistic; eq'n 5.4 of Wall and Jenkins
print(t) #Book says this should be 1.12, but here it's 1.33?? I do have an older edition of the book...

a = sigma_x**2/5
b = sigma_y**2/5
nu = 4*(a + b)**2/(a**2 + b**2) #A complicated way of getting the degrees of freedom; see Devore pg 357. W&J uses 5+5-2 ?
print(np.floor(nu))

f = scipy.stats.t.cdf(-t, 6) #integrate t-dist PDF from 0 to -t with nu = 6 dof
print(f)
print(scipy.stats.t.sf(t, 6)*2, '!') #p-value using above t; my own addition 


t, p = scipy.stats.ttest_ind(x, y, equal_var=False) #calculate t-test for means of two samples x and y, assuming non-equal variances

print(t) #scipy t-statistic; note it is the negative of the t-statistic above 
print(p) #scipy p-value; ~22% chance of these data being from the same mean

print(t*np.sqrt(a + b)) #+- confidence interval. 66% CI? See Devore page 358

#now find the 95% and 78% CI. We use the stats.t.cdf to get the t-value, corresponding to these levels 
print('\n---\n')
t_95 = scipy.stats.t.ppf(1-0.05/2, 6) #two-tailed CI
t_78 = scipy.stats.t.ppf(1-0.22/2, 6)

c = np.sqrt(a+b)
md = np.mean(y) - np.mean(x) #difference in means

ll_95,ul_95 = md-t_95*c, md+t_95*c
ll_78,ul_78 = md-t_78*c,md+t_78*c
print("The 95% CI is ({0:0.4f}, {1:0.4f})".format(ll_95,ul_95))
print("The 78% CI is ({0:0.4f}, {1:0.4f})".format(ll_78,ul_78))

