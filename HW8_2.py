# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:42:04 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

prior = 0.5
x = 0.5 #data
likelihood = lambda theta: 1/np.sqrt(2*np.pi) * np.exp(-(x-theta)**2/2) #sigma = 1, take advantage of only having 1 data point
evidence = prior*quad(likelihood, -1,1)[0]

post = lambda theta: likelihood(theta) * prior/evidence

theta_arr = np.linspace(-1,1,1000)
y_arr = post(theta_arr)
print(quad(post, -1,1)) #check that area under post is 1
plt.figure(figsize = (12,14))
plt.title('Posterior Distribution', size = 24)
plt.plot(theta_arr, y_arr, color = 'k', linestyle = '', marker = 'o')
plt.xlabel('$\\theta$', size = 18)
plt.ylabel('$P(\\theta | D)$', size = 18)
plt.grid()
plt.show()
