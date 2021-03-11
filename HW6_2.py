# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:46:55 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(23) #42 makes formula CI not capture histogram nicely

#set Parameters for problem
mu = 10
sigma = 1
n = 10
N = 1
nbin = 50

#define useful functions
def z_interval(xbar, s):
    '''
    Z-score 95% confidence interval for xbar
    '''
    ns = n#len(xbar)
    lbound = xbar-1.96 * s/np.sqrt(ns)
    ubound = xbar+1.96 * s/ np.sqrt(ns)
    return (lbound, ubound)

def t_interval(xbar, s):
    '''
    t-distribution 95% confidence interval for xbar
    '''
    ns = n#len(xbar)
    lbound = xbar-2.262 * s/np.sqrt(ns)
    ubound = xbar+2.262 * s/ np.sqrt(ns)
    return (lbound, ubound)

sample = np.random.normal(mu,sigma,(N,n))               #draw a sample of 10 numbers 

sxbar = np.average(sample)                              #Compute sample average
s = np.std(sample, ddof = 1)                            #Compute sample standard deviation

za2 = 1.96                                              #Z-score for 95% confidence

z_lbound = sxbar - za2 * sigma / np.sqrt(n)             #Compute lower and upper CI bounds with formula
z_ubound = sxbar + za2 * sigma/np.sqrt(n)

ta2 = 2.262                                             #Likewise for t-distribution

t_lbound = sxbar -ta2 * s/np.sqrt(n)
t_ubound = sxbar + ta2*s/np.sqrt(n)

verify_samp = np.random.normal(mu, sigma, (10000,n))    #draw 10 samples 10,000 times

verify_xbar = np.average(verify_samp, axis = 1)
verify_s = np.std(verify_samp, axis = 1, ddof = 1)

nv = len(verify_xbar)
z_arr = []                                              #store intervals for z and t distributions
t_arr = []

for i in range(nv):                                     #Compute 95% CI for z and t distributions for each xbar
    xbar = verify_xbar[i]
    s = verify_s[i]
    z_el = z_interval(xbar, sigma)
    t_el = t_interval(xbar, s)
    
    z_arr.append(z_el)
    t_arr.append(t_el)
    
zcount = 0                                              #counter for number of z and t intervals containing mu
tcount = 0

for i in range(nv):                                     #count how many intervals contain mu
    z_int = z_arr[i]
    t_int = t_arr[i]
    
    if z_int[0] <= mu <= z_int[1]:
        zcount+=1
    if t_int[0] <= mu <= t_int[1]:
        tcount+=1

print("{0:0.3f}% of intervals contain mu in the t-distribution".format(tcount/10000*100))
print("{0:0.3f}% of intervals contain mu in the z-distribution".format(zcount/10000*100))

#make figure
plt.figure(figsize = (12,14))
plt.title('95% Confidence Intervals for $\\overline{{X}}$  \
$({0:0.3f} \leq \\overline{{X}}_z \leq {1:0.3f})$, $({2:0.3f} \leq \\overline{{x}}_t \leq {3:0.3f})$'\
.format(z_lbound, z_ubound, t_lbound, t_ubound),size = 20)

bins = np.linspace(np.min(verify_xbar), np.max(verify_xbar), nbin+1)
plt.hist(verify_xbar, bins = bins, color = 'k', density=True)

plt.axvline(z_lbound, color = 'red', label = 'Z Dist Formula')                  #Show 95% CI from questions 1 & 2
plt.axvline(z_ubound, color = 'red')
plt.axvline(t_lbound, color = 'blue', label = 't Dist Formula')
plt.axvline(t_ubound, color = 'blue')

plt.axvline(np.mean([el[0] for el in t_arr]), color = 'pink', linestyle = '--') #Show 95% CI from bootstrap method
plt.axvline(np.mean([el[1] for el in t_arr]), label = 't Dist Bootstrap', color = 'pink', linestyle = '--')
plt.axvline(np.mean([el[0] for el in z_arr]), color = 'cyan', linestyle = '--')
plt.axvline(np.mean([el[1] for el in z_arr]), label = 'Z Dist Bootstrap', color = 'cyan', linestyle = '--')

plt.legend()
plt.xlabel('$\\overline{{X}}$',size = 18)
plt.ylabel('Probability Density', size = 18)
plt.show()

