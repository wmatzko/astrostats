# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:56:03 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(42)

#set parameters of problem
mu = 1
sigma = 1
n = 10                          #number of samples
N = 10000                       #number of trials
nbin = 50                       #number of bins in histogram

sample = np.random.normal(mu,sigma,(N,n))
sub_samp = [-0.546, -0.406, -0.115, -1.262, -1.386, -0.448, 0.829, 0.799, -1.100, 0.385]

Y = np.sum(np.square(sample), axis = 1)                             #Compute X1^2 + X2^2 +... for each trial

Y_draws = np.random.choice(sub_samp, (N,n), replace=True)           #draw 10 samples with replacement 10,000 times 
Ys = np.sum(np.square(Y_draws), axis = 1)                           #compute X1^2 + X2^2 for this new sample


bins = np.linspace(np.min(np.concatenate([Y, Ys])), np.max(np.concatenate([Y,Ys])), nbin+1)

#make histogram of probability densities
plt.figure(figsize = (12,14))
plt.title('Distribution of Y and Y*', size = 24)
pdf,_,_=plt.hist(Y, bins, density=True, alpha = 0.5, label = 'Y dist')
pdf_s,_,_=plt.hist(Ys, bins, density=True, alpha = 0.5, label = 'Y* dist')
plt.xlabel('Y', size = 18)
plt.ylabel('Probability Density', size = 18)
plt.legend()
plt.show()

#make scatter plot of probabiity densities

#pdf = sorted(pdf)                                                  #optional sorting of PDFs
#pdf_s = sorted(pdf_s)
plt.figure(figsize = (12,14))
plt.title('Y and Y* PDF Correlation', size = 24)
plt.plot(pdf, pdf_s, marker = 'o', color = 'k', linestyle = '')
plt.xlabel('Y PDF', size = 18)
plt.ylabel('Y* PDF', size = 18)
plt.grid()
plt.show()
