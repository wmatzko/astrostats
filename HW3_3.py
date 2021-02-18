# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 01:22:30 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt

#create array of probabilities for theta

theta = np.linspace(0,1,1000)

#define functions for P(theta|D) for each part of the problem
f1 = lambda x: 2*x
f2 = lambda x: 6*x*(1-x)
f3 = lambda x: 2 / 0.6551171 * x * (1-x) * np.exp(-(x - 0.5)**2/0.1)

#graph them

plt.figure(figsize = (12,14))
plt.title("Probability Curve with D = {H} and Flat Prior", size = 24)
plt.plot(theta, f1(theta), color = 'k', linestyle = '-')
plt.xlabel('$\\theta$', size = 18)
plt.ylabel('P($\\theta|D)$', size = 18)
plt.grid()
plt.show()

plt.figure(figsize = (12,14))
plt.title("Probability Curve with D = {H,T} and Flat Prior", size = 24)
plt.plot(theta, f2(theta), color = 'k', linestyle = '-')
plt.xlabel('$\\theta$', size = 18)
plt.ylabel('P($\\theta|D)$', size = 18)
plt.grid()
plt.show()

plt.figure(figsize = (12,14))
plt.title("Probability Curve with D = {H,T} and Gaussian Prior", size = 24)
plt.plot(theta, f3(theta), color = 'k', linestyle = '-')
plt.xlabel('$\\theta$', size = 18)
plt.ylabel('P($\\theta|D)$', size = 18)
plt.grid()
plt.show()