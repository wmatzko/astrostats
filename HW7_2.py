# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:20:15 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(623)

def find_n(n0, beta_true, tol=1E-9, niter = 10000):
    '''
    Finds the value of n for which beta_exp returns beta_true within specified tolerance
    Input:
        n0 - Initial guess for n
        beta_true - The desired beta value
        tol - Cutoff value (absolute error) for matching beta_true to beta_exp
        niter - Maximum number of iterations to perform
    Output:
        n - The value of n that makes beta_exp = beta_true within tol
        flag - Boolean; if true, successful in finding n
    '''
    n = n0
    #beta_exp_arr = [] #hold values of beta_exp
    for i in range(niter+1):        
        #print(i,n)
        #compute experimental beta probability
        beta_exp = beta_val(mu,mup,za,n)
        #print(i,n,beta_exp)
        #beta_exp_arr.append(beta_exp)
        if (abs(beta_exp - beta_true) > tol) and (i < niter):
                # This is fine; if speed was an issue, could use an
                # optimization method.
                n += (beta_exp-beta_true)/10 #crude way of updating n that seems to work for most cases
                
        elif (abs(beta_exp - beta_true) <= tol) and (i <niter):
            #print("Successfully found n after {0} iterations".format(i+1))
            flag = 1
            return n,flag
        elif i==niter:
            print("Warning, optimal n could not be found. Returning last iteration.")
            flag = 0
            return n,flag
        else:
            print('Other error')
            flag = 0
            return n,flag
        
def beta_val(mu,mup,za,n):
    '''
    Compute the two-tailed probability of beta(mu') at given alpha (from za)
    Input:
        mu - The true value of the mean
        mup - The alternative value of the mean
        za - Z-value of alpha should be positive
        n - Sample size
    Output:
        The Type II error probability beta(mu')
    '''
    
    return stats.norm.cdf(abs(za/2) + (mu-mup)/(sigma/np.sqrt(n)))  - stats.norm.cdf(-abs(za/2) + (mu-mup)/(sigma/np.sqrt(n)))#Book notation seems to take abs(z) here, or 1-alpha


mu = 1300
mup = 1320 #beta(mu'), probability of type II error at particular alternative value
sigma = 1.5 #assume known

alpha = 0.01 #probabiity of type I error
beta_true = 0.01 #desired Type II error probability at given point 


za = stats.norm.ppf(alpha)
zb = stats.norm.ppf(beta_true)


n_book = (sigma*(za/2+zb)/(mu-mup))**2 #minimum n needed to obtain beta_true, according to book
beta_exp = beta_val(mu,mup,za,n_book)  #Compute actual Type II error probability with n = n_book
beta_err = abs(beta_exp-beta_true)     #compute error

n_true,flag = find_n(n_book, beta_true)

if not flag:
    raise ValueError("Error: Could not find n. Results not reliable, cancelling script")
    
n_err = abs(n_true - n_book)

print("The approximate value of n is {0:0.5f}, while the true value is {1:0.5f}.\n The error is {2:0.3E}\n".format(n_book,n_true,n_err))
print("Beta(mu') with the approximate n is {0:0.3E}, while the true beta is {1:0.3E}.\n The error is {2:0.3E}\n".format(beta_exp, beta_true, beta_err))

#print(beta_val(mu,mup,za,n_true))
