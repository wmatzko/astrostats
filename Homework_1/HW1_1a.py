# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:56:53 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText as AT #to add fancy box of statistics
np.random.seed(42) #set random seed to generate same random data

mu = 0 #mean of Gaussian
sig = 1 #standard deviation of Gaussian

n = 100 #number of samples
niter = 10000 #number of iterations in for loop
nbin = 50 #number of bins in histogram

avgs = [] #hold the averages
for i in range(niter): #compute averages for however many iterations
    x = np.random.normal(mu, sig, size = n) #generate set of random numbers of size n from Gaussian dist.
    x_avg = np.average(x) #compute average
    avgs.append(x_avg) #add to list

bins = np.linspace(min(avgs), max(avgs), nbin+1) #make bins

plt.figure(figsize = (12,14)) #make figure
plt.title("Distribution of $\overline{X}$", size = 24)
plt.hist(avgs, bins, color = 'black')
plt.xlabel('Trial Averages', size = 18)
plt.ylabel('Frequency', size = 18)
#add box of possibly useful stats
# \overline{X}_dist is just the mean of the plotted distribution
txt = "$\mu = {0:0.2f}$\n$\sigma = {1:0.2f}$\n$\\overline{{X}}_{{\\mathrm{{dist}}}} = {2:0.2f}$\n$\\overline{{s}} = {3:0.2f}$\n$\\mathrm{{min}} = {4:0.2f}$\n$\\mathrm{{max}} = {5:0.2f}$\n$\\mathrm{{n}} = {6}$\n$\\mathrm{{niter}} = {7}$"\
.format(mu, sig, np.mean(avgs), np.std(avgs), min(avgs), max(avgs), n, niter)
pt = AT(txt, loc = 'upper left')
ax = plt.subplot(1,1,1) #Apparently this method will be depricated soon...not sure how else to add the label?
ax.add_artist(pt)
plt.grid()
plt.show()