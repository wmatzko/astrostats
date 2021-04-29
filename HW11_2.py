# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 05:56:52 2021

@author: William
"""
import numpy as np
import emcee
from scipy.stats import mode
from matplotlib import pyplot as plt
from scipy.stats import bayes_mvs
import corner

size = 50                       #number of data points in D
prior_range = 1                 #+- uniform hard bound

#choose what distribution to draw from
#D = np.random.rayleigh(0.5,size)
#D = np.random.normal(0.5, 1,size)
#D = np.random.uniform(-0.4,0.8, size)
D = np.random.chisquare(1, size)

print("The data are\n {0}\n".format(D))

walker_verbose = True           #if True, print statistics on individual walkers
nbin = 15

nstep = 1e3                     # Number of steps each walker takes
nwalk = 10                      # Number of initial values for theta
ndims = 1                       # Number of unknown parameters in theta vector

#scheme doesn't work well if nwalk%nrow != 0 
nrow = 5                        #number of rows for walker pdf grid plot
ncol = int(nwalk/5)             #number of columns

def auto_window(taus, c):
    '''
    I have no clue what this does. 
    Taken from emcee docs on autocorrelation analysis
    '''
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_new(y, c=5.0):
    '''
    Use the built-in autocorrelation analysis in emcee
    Taken from emcee docs
    '''
    f = np.zeros(y.shape[1])
    for yy in y:
        f += emcee.autocorr.function_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def log_LxP(theta, D):
    """Return Log( Likelihood * Posterior (prior?**) ) given data D."""
    p_arr = []
    for d in D:
        p = np.exp(-(theta-d)**2/2)
        p_arr.append(p)
    if np.abs(theta) <= prior_range:
        LxP = np.prod(p_arr)
    else:
        LxP = 0.0
    return np.log(LxP)

# Create a set of 10 inital values for theta. Values are drawn from
# a distribution that is unform in range [-1, 1] and zero outside.
thetas_initial =  np.random.uniform(-prior_range, prior_range, (nwalk, ndims))

# Initialize the sampler object
sampler = emcee.EnsembleSampler(nwalk, ndims, log_LxP, args=(D, ))

# Run the MCMC algorithm for each initial theta for 5000 steps
sampler.run_mcmc(thetas_initial, nstep, progress=True);

# Get the values of theta at each step
samples = sampler.get_chain()

ordered_samples = []                    #Reorganize chains so each list is one walker
#print(samples.shape) # (nstep, nawlk, ndims)

print() #console readability

if walker_verbose:
    for i in range(nwalk):
        el = samples[:,i,0]
        el_mean = np.mean(el)
        el_mode = mode(el)[0][0]
        el_median = np.median(el)
        el_ci = bayes_mvs(el)[0][1]
        err = np.percentile(el, [2.5,97.5]) #95 percentile of data
        std = np.std(el)
        ordered_samples.append(el)
        
        print("Walker {0} mean is {1:0.3f}".format(i+1,el_mean))
        print("Walker {0} median is {1:0.3f}".format(i+1,el_median))
        print("Walker {0} mode is {1:0.3f}".format(i+1,el_mode))
        print("Walker {0} standard deviation is {1:0.3f}".format(i+1,std))
        print("Walker {0} 95th percentile range is [{1:0.3f}, {2:0.3f}]".format(i+1, err[0], err[1]))
        print("Walker {0} Bayesian 90% CI is [{1:0.3f}, {2:0.3f}]\n".format(i+1, el_ci[0], el_ci[1]))

ordered_samples = np.array(ordered_samples)
flat_samples = np.ndarray.flatten(ordered_samples)
sample_percentile = np.percentile(flat_samples, [2.5,97.5])
sample_ci = bayes_mvs(flat_samples, alpha = 0.9)[0][1] #alpha is probability parameter is in interval

print("Across all walkers:")
print(" Sample mean is {0:0.3f}".format(np.mean(flat_samples)))
print(" Sample median is {0:0.3f}".format(np.median(flat_samples)))
print(" Sample mode is {0:0.3f}".format(mode(flat_samples)[0][0]))
print(" Sample standard deviation is {0:0.3f}".format(np.std(flat_samples)))
print(" Sample 95th percentile range is [{0:0.3f}, {1:0.3f}]".format(sample_percentile[0], sample_percentile[1]))
print(" Sample Bayesian 90% CI is [{0:0.3f}, {1:0.3f}]".format(sample_ci[0], sample_ci[1]))

#make plots

fig, axes = plt.subplots(nwalk, figsize = (12,20), sharex = True)
fig.suptitle("Parameter Values for Different Walkers", size = 24, y = 0.93)

for i in range(nwalk):
    ax = axes[i]
    ax.plot(samples[:,i,0], color = 'k')
    ax.set_xlim(0,nstep)
    ax.set_ylim(-prior_range-0.1, prior_range+0.1)
    yl = ax.set_ylabel("Walker {0}".format(i+1))
    yl.set_rotation(0)
    ax.yaxis.set_label_coords(-0.06, 0.4)
    
axes[-1].set_xlabel('Step Number', size = 18)

fig2, axes2 = plt.subplots(nrow, ncol, figsize = (14,14), sharex = 'col', sharey = 'row')
bins = np.linspace(np.min(flat_samples), np.max(flat_samples), nbin + 1)
pcount = 0
fig2.suptitle("PDFs for Different Walkers", size = 24, y = 0.93)
for i in range(5):
    for j in range(int(np.ceil(nwalk/5))):
        ax = axes2[i,j]
        ax.hist(samples[:,pcount,0], bins = bins, color='k', density = True, label = 'Walker {0}'.format(pcount+1))
        #ax.set_xlabel('$\\theta$')
        ax.legend()
        pcount+=1
    
for i in range(ncol):
    axes2[-1,-i-1].set_xlabel('$\\theta$')
for i in range(nrow):
    axes2[i,0].set_ylabel('P($\\theta | D$)')
    
    
#do autocorrelation analysis and plot it
    
#create evenly spaced points in log space to sample with
N = np.logspace(2, np.log10(nstep), 10, dtype = int)#np.exp(np.linspace(np.log(100), np.log(nstep), 10)).astype(int)
auto_arr = np.full(len(N), np.nan)

for i,n in enumerate(N):
    auto_arr[i] = autocorr_new(ordered_samples[:,:n])
    
plt.figure(figsize = (12,14))
plt.title('Autocorrelation Analysis', size = 24)
plt.loglog(N, auto_arr, color = 'r', marker = 'o', linestyle = '-', label = 'emcee built-in analysis')
plt.plot(N, N/50, color = 'k', marker = '', linestyle = '--', label = '$\\tau = N/ 50$')
ylim = plt.gca().get_ylim()
plt.ylim(ylim)
plt.xlabel("Number of samples, $N$", size = 18)
plt.ylabel("$\\tau$ estimates", size = 18)
plt.grid()
plt.legend()
plt.show()

#Also make corner plot because why not...there's actually only 1 parameter so there's good reason not to but whatever...
fig = corner.corner(flat_samples, labels = ['$\\theta$'], quantiles = [0.16,0.50,0.84], show_titles = True) #basically the same as the histograms above for 1D data
#print(np.percentile(flat_samples, [16,50,84]))