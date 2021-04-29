import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mode
from scipy.stats import bayes_mvs
import corner
N = 10000
verbose = False #print output diagnostics

if False:
    D = [0.0]
    dmu = 0.5
    mu = np.arange(-5, 5 + dmu, dmu)
    P = np.exp( -(mu**2)/2 )/np.sqrt(2*np.pi)
    sample = np.random.normal(0, 1, N)

if True:
    D = [0.5, 1.5]
    dmu = 0.05
    #mu = np.arange(-3, 4 + dmu, dmu)
    mu = np.arange(-1, 1 + dmu, dmu)

    p1 = np.exp( -((mu-D[0])**2)/2 )/np.sqrt(2*np.pi)
    p2 = np.exp( -((mu-D[1])**2)/2 )/np.sqrt(2*np.pi)
    
    P = p1*p2
    P = P/np.sum(P*dmu)

    sample = np.random.normal((D[0] + D[1])/2, 1/np.sqrt(2), N)


i_mu = 1 # Initial index for mu

step_size = 1

# Algorithm:
#
#   1. First opinion fair coin is heads
#      1a. If P_right > P_curr, step right
#      1b. If P_right <= P_curr, get second opinion:
#          Flip biased coin with p_heads = P_right/P_curr.
#          If heads, follow first opinion.
#          If tails, don't step.
#
#   2. First opinion coin is tails
#      1a. If P_left > P_curr, step left
#      1b. If P_left <= P_curr, get second opinion
#          If P_right <= P_curr, get second opinion:
#          Flip biased coin with p_heads = P_left/P_curr.
#          If heads, follow first opinion.
#          If tails, don't step.

# History of steps
i_mu_hist = np.zeros(N, dtype=np.int64)
for i in range(N):

    # "First opinion" fair coin toss
    ht = np.random.binomial(1, 0.5)

    P_right = P[i_mu+1]
    P_curr = P[i_mu]
    P_left  = P[i_mu-1]

    if ht == 1:
        flip1 = "H"
        if P_right > P_curr:
            # If first opinion coin toss says step right and P is higher to right, step right
            flip2 = "N/A" # Not Applicable
            step_dir = 1
        else:
            # If first opinion coin toss says step right and P is lower to right, flip biased
            # coin with p_heads = P_right/P_curr as second opinion.
            ht = np.random.binomial(1, P_right/P_curr)
            if ht == 1:
                flip2 = "H"
                # Second opinion "H" means follow first opinion instructions
                step_dir = 1
            else:
                flip2 = "T"
                # Don't take a step
                step_dir = 0
    else:
        flip1 = "T"
        if P_left > P_curr:
            # If first opinion coin toss says step left and P is higher to left, step left
            flip2 = "N/A"
            step_dir = -1
        else:
            # If first opinion coin toss says step left and P is lower to left, flip biased coin
            # with p_heads = P_left/P_curr for second opinion.
            ht = np.random.binomial(1, P_left/P_curr)
            if ht == 1:
                flip2 = "H" 
                # Second opinion "H" means follow first opinion instructions
                step_dir = -1
            else:
                flip2 = "T"
                # Don't take a step
                step_dir = 0

    # Take step
    if step_dir != 0:
        i_mu = i_mu + step_size*step_dir

    # Handle steps out-of-bounds by not taking a step
    if i_mu < 1:
        i_mu = 0
    if i_mu > mu.shape[0] - 2:
        i_mu = mu.shape[0] - 2

    i_mu_hist[i] = i_mu
    
    if verbose:
        if i < 100 or np.mod(i, 100) == 0:
            print('i = {0:3d}; step = {1:2d} P_left = {2:.5f}; P_curr = {3:.5f}; '\
                  .format(i, step_dir, P_left, P_curr), end="")
            print('P_right = {0:.5f}; Flip 1: {1:s}; Flip 2: {2:s}' \
                  .format(P_right, flip1, flip2))

# Remove first 10% of history of steps
a = np.int(np.round(0.1*N))
i_mu_hist_r = i_mu_hist[a:]

data = mu[i_mu_hist_r]

data_percentile = np.percentile(data, [2.5,97.5]) #95th percentile range
data_ci = bayes_mvs(data, alpha = 0.9)[0][1] #90% probability that mean is in this range

print("Mean of hist is {0:0.3f}".format(np.mean(data)))
print("Median of hist is {0:0.3f}".format(np.median(data)))
print("Mode of hist is {0:0.3f}".format(mode(data)[0][0]))
print("Standard deviation of hist is {0:0.3f}".format(np.std(data)))
print("The 95th percentile range is [{0:0.3f}, {1:0.3f}]".format(data_percentile[0], data_percentile[1]))
print("The Bayesian 90% CI is [{0:0.3f}, {1:0.3f}]".format(data_ci[0], data_ci[1]))

if N <= 10000:
    plt.figure()
    plt.axvline(a, c='k', ls='--', label='burn-in step')
    plt.plot(mu[i_mu_hist])
    plt.grid()
    plt.xlabel('Step index, i_mu')
    plt.ylabel('$\mu$ associated with i_mu')

plt.figure()
plt.grid()
#plt.xlim([-1, 1])
#plt.ylim([0, 0.8])
bins = 50
bins = mu - dmu/2 # Center bins on possible x positions
plt.hist(mu[i_mu_hist_r], bins=bins, density=True, histtype='step', 
         label='Metropolis sampling')
plt.hist(sample, bins=bins, density=True,  histtype='step',
         label='Direct sampling from Exact')
plt.plot(mu, P, 'k',label='Exact ($\mathcal{N}(\mu=1, \sigma^2=1/2)$)')
plt.ylabel('$p(\\theta|\\mathcal{D})$')
plt.xlabel('$\\theta$')
plt.legend(loc='upper left')
plt.title('N = {0:d}; $\mathcal{{D}}=[0.5, 1.5]$'.format(N))
plt.show()

fig = corner.corner(data, quantiles = [0.16,0.50,0.84], show_titles = True)