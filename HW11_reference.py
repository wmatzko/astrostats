import numpy as np
import emcee
from scipy.stats import mode
from matplotlib import pyplot as plt
from scipy.stats import bayes_mvs
import corner

D = [0.5, 1.5]

def log_LxP(theta, D):
    """Return Log( Likelihood * Posterior) given data D."""
    p1 = np.exp( -((theta-D[0])**2)/2 )
    p2 = np.exp( -((theta-D[1])**2)/2 )
    if np.abs(theta) <= 1:
        LxP = p1*p2
    else:
        LxP = 0.0
    return np.log(LxP)

nstep = 1e3     # Number of steps each walker takes
nwalk = 10      # Number of initial values for theta
ndims = 1       # Number of unknown parameters in theta vector

# Create a set of 10 inital values for theta. Values are drawn from
# a distribution that is unform in range [-1, 1] and zero outside.
thetas_initial =  np.random.uniform(-1, 1, (nwalk, ndims))

# Initialize the sampler object
sampler = emcee.EnsembleSampler(nwalk, ndims, log_LxP, args=(D, ))

# Run the MCMC algorithm for each initial theta for 5000 steps
sampler.run_mcmc(thetas_initial, nstep, progress=True);

# Get the values of theta at each step
samples = sampler.get_chain()
ordered_samples = []
#print(samples.shape) # (nstep, nawlk, ndims)

print()

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

plt.figure()
plt.plot(samples[:,0,0])
plt.title('First Walker')
plt.xlabel('Step')
plt.ylabel('$\\theta$')
plt.grid()
plt.show() #bothers me that this isn't here

plt.figure()
plt.hist(samples[:,0,0])
plt.title('Histogram of $\\theta$ values for first walker')
plt.ylabel('# in bin')
plt.xlabel('$\\theta$')
plt.grid()
plt.show()

plt.figure()
plt.hist(samples[:,0,0],density=True)
plt.title('pdf of $\\theta$ values for first walker')
plt.ylabel('p($\\theta|\\mathcal{D})$')
plt.xlabel('$\\theta$')
plt.grid()
plt.show()

fig = corner.corner(flat_samples, labels = ['$\\mu$'], quantiles = [0.16,0.50,0.84], show_titles = True) #basically the same as the histograms above for 1D data
