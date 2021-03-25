# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 03:32:28 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error 
np.random.seed(42)

def linear_regression(x,y):
    '''
    Function to perform linear regression on y = bx + a
    '''
    
    b = np.cov(x,y, bias = True)[0][1]/np.var(x) #Nicer formula; not sure why need bias=True to match SciPy's results, since that flag uses ddof = 0 not 1
    a = np.mean(y) - b*np.mean(x)
    
    return b,a

N = 1000                               #number of data points in full sample
Nb = 10000                             #number of bootstrap samples
n = 20                                 #number of data points in each sub-sample
nbin = 30                              #number of bins for histogram
mu = 0
sigma = 0.2

alpha = 1
beta = 1
yt = lambda x, eps: beta*x + alpha + eps             #'true' function

xval = np.linspace(0,1, N, endpoint =  False)        #Create desired x values
yval = np.nan*np.zeros(N)

#make y values with noise
for i in range(N):
    rand = np.random.normal(mu,sigma)
    el = yt(xval[i],rand)
    yval[i] = el
    
pairs = np.array(list(zip(xval,yval)))                #create ordered pairs for data
idx = np.random.randint(0,N, size = n)                #generate random indices to select n data from pairs
npoint = pairs[idx]

xn,yn = [x[0] for x in npoint], [x[1] for x in npoint] #extract x and y data
xn,yn = np.array(xn),np.array(yn)

#Do curve fitting with selected data
my_b,my_a = linear_regression(xn,yn)                   #Using my function
popt, pcov = curve_fit(lambda x,b,a: b*x + a, xn,yn)   #Using SciPy
sci_b, sci_a = popt[0],popt[1]
sci_err = np.sqrt(np.diag(pcov))                       #1-sigma error bars, for comparison

y_fit = lambda x: my_b * x + my_a                      #make function for line fit 
y_fit_eval = y_fit(xval)

#Calculate 10,000 values for b and a with 20 draws each iteration

idx = np.random.randint(0,N,size=(Nb,n))               #random indicies to draw
bsample = pairs[idx]

b_arr, a_arr = np.nan*np.zeros(Nb),np.nan*np.zeros(Nb)
for i in range(Nb):
    xb,yb = [x[0] for x in bsample[i]], [x[1] for x in bsample[i]]
    my_b_el,my_a_el = linear_regression(xb,yb)
    b_arr[i] = my_b_el
    a_arr[i] = my_a_el
    
b_low,b_med,b_high = np.percentile(b_arr, [2.5,50,97.5]) #obtain 95th percentile and median
a_low,a_med,a_high = np.percentile(a_arr, [2.5,50,97.5])
    
#make plot of simulated data and bootstrap CI
plt.figure(figsize = (12,14))
plt.title('Simulated Data', size = 24)
plt.plot(xval,yval, marker = 'o', linestyle = '', color = 'k', label = 'Full Data')
plt.plot(xn,yn, marker = 'o', color = 'red', linestyle = '', markerfacecolor = 'none', markeredgewidth = 1,label = 'Selected Data')
plt.plot(xval,y_fit_eval, marker = '', color = 'aqua', linestyle = '-', label = 'Best Fit Line')
plt.fill_between(xval, b_low*xval + a_low, b_high*xval+a_high, label  = 'Bootstrap 95% CI', alpha = 0.5, color = 'teal')
plt.xlabel('x', size = 18)
plt.ylabel('y', size = 18)
plt.grid()
plt.legend()
plt.show()

print("My computed slope is b = {0:0.5f} and y-intercept is a = {1:0.5f}\n".format(my_b,my_a))
print("SciPy's computed slope is b = {0:0.5f} and y-intercept is a = {1:0.5f}\n".format(sci_b,sci_a))
    
#Plot histograms for a and b
    
plt.figure(figsize = (12,14))
plt.title('Distribution of Slope',size = 24)
bins= np.linspace(np.min(b_arr), np.max(b_arr), nbin+1)
plt.hist(b_arr, bins = bins, color = 'k')
plt.axvline(b_low, color = 'red', linestyle = '--')
plt.axvline(b_med, color = 'red', linestyle = '-', label = '95% CI')
plt.axvline(b_high, color = 'red', linestyle = '--')
plt.xlabel('b',size = 18)
plt.ylabel('Frequency', size = 18)
plt.legend()
plt.show()

plt.figure(figsize = (12,14))
plt.title('Distribution of Y-Intercept',size = 24)
bins= np.linspace(np.min(a_arr), np.max(a_arr), nbin+1)
plt.hist(a_arr, bins = bins, color = 'k')
plt.axvline(a_low, color = 'red', linestyle = '--')
plt.axvline(a_med, color = 'red', linestyle = '-', label = '95% CI')
plt.axvline(a_high, color = 'red', linestyle = '--')
plt.xlabel('a',size = 18)
plt.ylabel('Frequency', size = 18)
plt.legend()
plt.show()

print("The bootstrap 95% CI for the slope is ({0:0.5f}, {1:0.5f})".format(b_low,b_high))
print("The bootstrap 95% CI for the y-intercept is ({0:0.5f}, {1:0.5f})\n".format(a_low,a_high))

print("SciPy's 1-sigma error for the slope is {0:0.5f}".format(sci_err[0]))
print("SciPy's 1-sigma error for the y-intercept is {0:0.5f}\n".format(sci_err[1]))

#Use t-test to find 95% CI. Adapted from https://online.stat.psu.edu/stat501/lesson/2/2.1 with n-2 dof
p = 0.05
t = stats.t.ppf(1-p/2, n-2)

yhat,ytrue = y_fit(np.array(xn)),yn

MSE = mean_squared_error(ytrue,yhat)
ssq = np.sum([(x - np.mean(xn))**2 for x in xn])
den = np.sqrt(ssq)
b_err = t * np.sqrt(MSE)/den
b_CI = (my_b - b_err, my_b + b_err)

print("The 95% CI on the slope from the t-test is ({0:0.5f}, {1:0.5f})".format(b_CI[0], b_CI[1]))

a_err = t*np.sqrt(MSE) *np.sqrt(1/n + np.mean(xn)/ssq)

a_CI = (my_a - a_err, my_a + a_err)

print("The 95% CI on the slope from the t-test is ({0:0.5f}, {1:0.5f})".format(a_CI[0], a_CI[1]))

b_resid = a_arr-alpha
a_resid = b_arr - beta

plt.figure(figsize = (12,14))
plt.title('Residual Correlation', size = 24)
plt.plot(b_resid,a_resid, marker = 'o',color = 'k', linestyle = '')
plt.xlabel('Slope Residuals', size = 18)
plt.ylabel('Y-Intercept Residuals', size = 18)
plt.grid()
plt.show()
