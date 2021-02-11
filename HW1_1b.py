# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:56:53 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText as AT #to add fancy box of statistics
np.random.seed(42) #set random seed to generate same random data

#take same code from part a...don't know the point in splitting the scripts up...
# Easier to share and grade, but correct, bad style.

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
ax = plt.gca() #Apparently this method will be depricated soon...not sure how else to add the label?
ax.add_artist(pt)
plt.grid()
plt.show()

#how many of avgs were in range +- 0.01?

cut = [i for i in avgs if -0.01 <= i <= 0.01]
print("There are {0} averages in the range [-0.01, 0.01] for n = {1} and niter = {2}\n".format(len(cut), n, niter))

#if n increases, then larger fraction of x_bar is in the range +- 0.1
#if n decreases, then smaller fraction of x_bar is in range of +- 0.1
#this is easily verified by changing n and looking at the console output


#what is range containing 99% of data?

eps = 0.2652 #trial and error to find range for n = 100. 
#epst = np.percentile(avgs, 99) # good estimate of 99th percentile, but since data isn't exactly Gaussian there is some error. This only captures ~98% of data
cut = [i for i in avgs if -eps <= i <= eps] 
print("For n = {0} and niter = {1}, the range [-{2}, {2}] covers {3}% of the data\n".format(n, niter, eps, len(cut)/niter * 100))

#if n increases, then the same value of epsilon will encompass more than 99% of the data
# Hence, if n increases, epsilon should be smaller to cover 99% of the data

#if n decreases, the same epsilon will encompass less than 99% of the data
# Hence, if n decreases, epsilon should be larger to cover 99% of the data

# Again, the above is easily verified by changing n and looking at console output 

# How do the answers change if the distribution changes?
# I assume this means 'what happens if we change mu and sigma?'
# For simplicity and ease of comparison, let's just tweak sigma and see how things change 

# If Sigma is 10, then for n = 100 and niter = 10000, then 81/10000 (~0.8%) averages are in [-0.01, 0.01]
# An epsilon of 0.2652 only covers ~21% of the data (so we should seek a larger epsilon to cover 99%)

# If we decrease sigma to 0.5, then for n = 100 and niter = 10000 there are 1571/10000 (~16%) averages in [-0.01, 0.01]
# An epsilon of 0.2652 covers 100% of the data, so we of course should seek a smaller epsilon to encompass 99% of the data

