# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:53:57 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import itertools
from collections import Counter, OrderedDict
np.random.seed(45) #sometimes the simulation is accurate, sometimes it's not..

D = [0.5, 1.5] #data given in problem 
x1,x2 = D[0],D[1]
mu_est = np.mean(D) #best guess mu 
sigma = 1 #given in problem 
ll,ul = -1,1 #lower and upper limits to plot

likelihood = lambda x1,x2: 1/(2*np.pi) * np.exp(-0.5*((x1-mu_est)**2 + (x2-mu_est)**2))

#Prior is some constant = c
c = 1/(ul-ll) #from integrating p(theta)dtheta from ll to ul and making that equal to 1. 

#evidence will be product of two integrals 

e_int = dblquad(likelihood, ll, ul, ll, ul)[0]
evidence = c**2 * e_int

#constant in the prior will cancel with the evidence 
post = lambda x1,x2: c**2 * likelihood(x1,x2) / evidence

xvals = np.linspace(ll,ul,10000) #xvalues for analytical function
post_arr = post(xvals,xvals) #plotting array for analytical post

check = dblquad(post, ll, ul, ll, ul) #check post integrates to 1
print(check)

#now do experimental method
N = 50000 #number of draws for a given mu
nbin = 20 #number of plotting bins; implicitly specifies number of mu values to use. 

mu_test = np.linspace(ll,ul, nbin+1) #test these values of mu
mu_arr = []
val_arr = []

for mu in mu_test:
    exp_vals = np.random.normal(mu, sigma, size = (N, 2))
    mu_arr.append(N*[mu]) #each output is assciated with a particular mu
    val_arr.append(exp_vals)
    
mu_arr = np.ndarray.flatten(np.array(mu_arr))

xdata = list(itertools.chain.from_iterable([x[:,0] for x in val_arr]))
ydata = list(itertools.chain.from_iterable([x[:,1] for x in val_arr]))

plt.figure(figsize = (12,14))
plt.title('2D Histogram of Generated Data', size = 24)
join = np.concatenate([xdata,ydata])
bins = np.linspace(np.min(join), np.max(join), nbin+1)
h,xedge,yedge,_ = plt.hist2d(xdata,ydata, bins=bins, density = True)
plt.xticks(bins, rotation = 90)
plt.yticks(bins)
plt.xlabel('x1 (first data point)', size = 18)
plt.ylabel('x2 (second data point)', size = 18)
plt.grid()
plt.colorbar()
plt.show()

xmask = [(xedge[i] < x1 and x1<=xedge[i+1]) for i in range(len(xedge)-1)].index(True) #find the location of the two bin edges trapping 0.5
x1_bin = xedge[xmask],xedge[xmask+1] #Find the actual two bin values trapping 0.5

ymask = [(yedge[i] < x2 and x2<=yedge[i+1]) for i in range(len(yedge)-1)].index(True) #repeat for y
x2_bin = yedge[ymask],yedge[ymask+1]

x_data_bin = np.array([x1_bin[0] < el <= x1_bin[1] for el in xdata]) #gather all the x-data in the 0.5 bin
y_data_bin = np.array([x2_bin[0] < el <= x2_bin[1] for el in ydata]) #gather all the y-data in the 1.5 bin

data_mask = x_data_bin&y_data_bin #find all data that's in both the 0.5 and 1.5 bin 
bin_mu = mu_arr[data_mask] #find all the corresponding mu values for that data

mu_dist = Counter(bin_mu) #tally up those mu values

for el in mu_test:
    if el not in list(mu_dist.keys()):
        mu_dist[el] = 0 #if a theta value in mu_test is not in counter, add it in and make it zero
        
mu_dist = OrderedDict(sorted(mu_dist.items())) #I like sorted dictionaries 
mu_norm = np.sum(list(mu_dist.values())) * (xedge[1] - xedge[0]) *evidence/c**2 #I might have specified this wrong, but it usually works OK...
#print(mu_dist)
print(mu_norm)

mu_exp_ocr = np.array(list(mu_dist.values()))/mu_norm  #normalized mu occurrence in bin 0.5,1.5

plt.figure(figsize = (12,14))
plt.title('Posterior Distribution', size = 24)
plt.plot(xvals, post_arr, marker = 'o', linestyle = '', color = 'k', label = 'Analytical')
plt.plot(mu_test, mu_exp_ocr,linestyle = '-', color = 'r', label = 'Experimental', drawstyle = 'steps-post')
plt.xlabel('$\\theta$', size = 18)
plt.ylabel('Posterior $P(\\theta | D )$', size = 18)
plt.grid()
plt.legend()
plt.show()