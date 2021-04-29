# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:21:57 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm

#set basic stuff
#D = [0.5, 1.5]                  #data given in problem 
D = np.random.normal(0.5,1,50)  #other data set to use

sigma = 1                       #Gaussian sd, given in problem 
ll,ul = -1,1                    #Lower and upper limits to do calculation on
c = 1/(ul-ll)                   #from integrating p(theta)dtheta from ll to ul and making that equal to 1. 
g_mu = 0.5                      #Gaussian mu for prior belief
g_sigma = 1                     #Gaussan sigma for prior belief

iters = 10000                   #Number of MCMC iterations
walkers = 3                     #Number of walkers
mu0 = 0.5                       #Initial guess
jump_width = 0.3                #Step size
burnin = 100                   #Burnin period
nbin = 20                       #number of bins to plot
cc = True                       #Test for convergence?
cc_min = 2000                   #Minimum iteration to start convergence test
cc_test = 500                   #Test for convergence every n iterations

#define useful functions
def Gaussian(x, mu = g_mu, sigma = g_sigma):
    '''
    PDF of a gaussian
    '''
    g = 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-0.5 * ( (x-mu)/ sigma )**2)
    return g

def flat(d,c=c):
    '''
    Function that returns constant value c. 
    d is a dummy index...I need a better way to specify general priors...
    '''
    return c

def calc_posterior_analytical(data, x, mu_0, sigma_0):
    '''
    Shamelessly stolen for comparative purposes 
    '''
    sigma = 1.
    n = len(data)
    mu_post = (mu_0 / sigma_0**2 + sum(data) / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
    sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
    return norm(mu_post, np.sqrt(sigma_post)).pdf(x)

def mcmc_sampler(data, mu0, prior, iters = 5000, walkers = 3, jump_width = 0.8,\
                 burnin = 500, cc = True, cc_min = 500, cc_test = 500):
    '''
    Use MCMC to compute a posterior distribution for a single variable mu assuming Gaussian distributions
    Input:
        Data - Values of mu from experiments
        mu0 - Initial guess for mu, single value
        Prior - Should be Python function with first argument as independent variable, 
                with other parameterizations specified by defailt in the func name
        iters - Numer oc MCMC iterations to perform
        walkers - Number of chains with different starting places
        jump_width - How wide our 'search radius' should be for proposed mu values
        burnin - Discard the first n iterations in calculation of val
        cc - If True, do converence check
        cc_min - Minimum number of iterations to perform before start checking for convergence
        cc_test - How often to check for convergence (e.g. every 500 iterations)
    Output:
        iter_arr - Index of each iteration; useful for plotting
        Post - Array of calculated posterior values
        Val - Mean parameter value from all chains
    '''
    #sanity checks
    if burnin >= iters:
        raise ValueError("Burn-in cannot be greater than the number of iterations")
    elif (walkers < 1) or not (isinstance(walkers, int)):
        raise ValueError("The number of walkers must be an integer greater than or equal to 1")
    elif cc_min < burnin:
        raise ValueError("You should not start checking for convergence inside the burn-in period")
    
    nwalk = walkers-1                               #first walker corresponds to original mu0
    mu0_w = [mu0,*list(norm.rvs(mu0, 1, nwalk))]    #generate new initial guesses centered around mu0
    post = [[el] for el in mu0_w]
    current_mu = np.ndarray.flatten(np.array(post)) #create currenet mu values from flattened post
    
    for i in range(iters):
        if not (i+1)%1000:
            print("Working on iteration {0}".format(i+1)) #crude counter
        
        #generate proposal, follow Gaussian structure 
        proposal = [norm(j, jump_width).rvs() for j in current_mu]

        #compute likelihood of current mu and proposed mu
        likelihood_current = [np.prod(norm(j, 1).pdf(data)) for j in current_mu]
        likelihood_proposal = [np.prod(norm(j, 1).pdf(data)) for j in proposal] 
        
        #Calculate priors. Use flat prior unlike in webpage example. 
        prior_current = [prior(j) for j in current_mu]
        prior_proposal = [prior(j) for j in proposal]
        
        #compute probabilities of current and proposal
        p_current = np.array(likelihood_current) * np.array(prior_current)
        p_proposal = np.array(likelihood_proposal) * np.array(prior_proposal)
        
        p_accept = p_proposal/p_current #acceptance probability
        
        accept = np.random.rand() < p_accept #acceptance criteria

        #Update chains
        for j in range(walkers):
            if accept[j]:
                current_mu[j] = proposal[j]
            post[j].append(current_mu[j])
            
        #Check for convergence...don't think GR statistic is very good, at least not here...
        if (cc) and (i>cc_min) and ( not (i+1)%cc_test):
            print("\nTesting for convergence at iteration {0}\n".format(i+1))
            conv, R = GR(post, burnin)
            if conv:
                print("Chains converged at {0} iterations (R = {1:0.3f})\n".format(i+1, R))
                break
            else:
                print("Chains have not converged at {0} iterations (R = {1:0.3f})\n".format(i+1, R))
                
    iter_arr = np.arange(0,i+2)                 #return number of iterations
    bchain = [j[burnin:] for j in post]         #discard first burnin elements in chains
    val = np.nanmean(bchain)                    #mean of bchain
    
    
    return iter_arr,np.array(post), val

#likelihood = lambda theta: 1/(2*np.pi) * np.exp(-0.5*((x1-theta)**2 + (x2-theta)**2))
def likelihood(theta, D):
    '''
    Likelihood function for n Gaussian data points assuming sigma = 1
    theta - Mean 
    D - Data
    '''
    #pref = 1/(2*np.pi)**(n/2)
    gauss = np.prod([Gaussian(theta, i, 1) for i in D])
    
    return gauss

def ACF(data, k):
    '''
    Compute autocorrelation function for data and lag time k
    '''
    r_arr = [] #hold ratio of sk_s0 for each chain
    m= len(data) #number of chains to compute ACF of
    n = len(data[0]) #length of chain, assume chains are identical length
    def s_k(k):
        sk = 1/n * np.sum([(y[j] - ybar) * (y[j-k] - ybar) for j in range(n-k)])
        return sk
    
    for i in range(m):
        y = data[i]
        ybar = np.mean(y)
        
        sk = s_k(k)
        s0 = s_k(0) 
        
        r = sk/s0 #this can sometimes be negative?
        
        r_arr.append(r)
        
    return r_arr

def GR(data, burnin):
    '''
    Calculates Gelman-Rubin statistic in attempt to assess chain convergence 
    '''
    data = [j[burnin:] for j in data] #discard burnin steps
    M = len(data) #number of chains
    N = len(data[0]) #length of chains, assume they are all same size
    post_mean = np.mean(data) #mean of all chains
    B = N/(M-1) * np.sum([(np.mean(data[j]) - post_mean)**2 for j in range(M)]) #between-chain variances
    W = 1/M * np.sum([np.var(data[j]) for j in range(M)]) #within-chain variances
    
    V = (N-1)/N * W + (M+1)/(M*N) * B
    
    R = np.sqrt(V/W)
    
    conv = R <= 1.1 #want R to be <= 1.1 to converge...roughly..
    return conv, R

def ESS(data):
    '''
    Computes the effective sample size and MCMC standard error
    '''
    ess_arr = [] #hold effective sample size for each chain
    mcse_arr = [] #MCMC standqrd deviation for each chain
    n = len(data[0]) #chain length, assume they're all the same size
    acf = ACF(data, 100)
    for i in range(len(data)):
        sd = np.std(data[i]) #standard deviation of chain
        ess = n/(1+2*acf[i])
        mcse = sd / np.sqrt(ess)
        ess_arr.append(ess)
        mcse_arr.append(mcse)
    
    return np.array(ess_arr), np.array(mcse_arr)

#create analytial solution, realize my solution in HW9 was nonsense 

#evidence will be product of n integrals, n is number of data points
e_int = quad(likelihood, ll,ul, args = (D))[0]#dblquad(likelihood, ll, ul, ll, ul)[0]
evidence = c**2 * e_int

#constant in the prior will cancel with the evidence 
post = lambda theta, D: c**2*likelihood(theta, D)/evidence

xvals = np.linspace(ll,ul,10000) #xvalues for analytical function
post_arr = [post(i, D) for i in xvals] #plotting array for analytical post

#make other analytical solution as described in twiecki
#post_analytical = calc_posterior_analytical(D, xvals, np.mean(D), 1)

#now do mcmc to estimate the posterior 
print("Performing MCMC with {0} iterations and {1} walkers\n".format(iters, walkers))
iter_arr, chains, val= mcmc_sampler(data = D, mu0 = mu0, prior = flat, iters = iters, walkers = walkers, jump_width = jump_width, burnin = burnin,\
                                    cc = cc, cc_min = cc_min, cc_test = cc_test)

print("The mean value of the chains is {0:0.4f}\n".format(val))

#now find ESS and MCMC standard error

ess,mcmc_se = ESS(chains)


#now do plotting

plt.figure(figsize = (12,14))
plt.title('Posterior, {0} iterations'.format(len(chains[0]) -1), size = 24)
plt.plot(xvals,post_arr, label = 'Analytical', color = 'k', marker = 'o', linestyle = '')
#plt.plot(xvals, post_analytical, label = 'Analytical_alt', color = 'gray', marker = 'o', linestyle = '')
for i in range(walkers):
    plt.hist(chains[i], bins = np.linspace(np.min(chains[i]), np.max(chains[i]),nbin+1), label = 'MCMC Chain {0}, ESS = {1:0.2f}'.format(i+1, ess[i]), density = True, rwidth=0.8, histtype = 'step')
plt.xlabel('$\\theta$',size = 18)
plt.ylabel('P($\\theta$ | D)', size = 18)
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize = (12,14))
plt.title('Trace Plot', size = 24)
for i in range(walkers):
    plt.plot(iter_arr, chains[i], label = 'MCMC Chain {0}, SE = {1:0.4f}'.format(i+1, mcmc_se[i]), marker = '', linestyle = '-' )
plt.axvline(burnin, color = 'red', linestyle = '-', label = 'Burnin', linewidth = 6)
plt.plot(iter_arr, np.mean(chains, axis = 0), label = 'Chains Mean', color = 'k', linestyle = '-', linewidth = 4)
plt.xlabel('Iteration',size = 18)
plt.ylabel('Param. Value', size = 18)
plt.grid()
plt.legend()
plt.show()

k_arr = np.arange(1,101) #generate lags
acf_arr = []
for i in range(len(k_arr)):
    acf_arr.append(ACF(chains, i))
acf_arr_sorted = []
for i in range(walkers):
    acf = [y[i] for y in acf_arr]
    acf_arr_sorted.append(acf)
    
plt.figure(figsize = (12,14))
plt.title('Autocorrelation Functions for MCMC Chains', size = 24)
for i in range(walkers):
    plt.plot(k_arr, acf_arr_sorted[i], label = "MCMC Chain {0}".format(i+1))
plt.xlabel('Lag (k)', size = 18)
plt.ylabel('ACF(k)', size = 18)
plt.grid()
plt.legend()
plt.show()


