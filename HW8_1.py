# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:53:02 2021

@author: William
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.integrate import quad
from scipy.optimize import dual_annealing as da
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm

#no points for efficiency...
def CI_est(f, a_min, b_max, step = 0.001, alpha = 0.05, tol = 0.001, brute = False):
    '''
    Use brute force or Dual Annealing to calculate the smallest 95% credible interval
    Input:
        f - pdf, or funtion to integrate
        a_min - Minimum bound
        b_max - Maximum bound
        step - Spacing between [a,b]; sub-intervals to sample
        tol - Acceptable tolerance for 95% area under f
        brute - If true uses brute force, else uses Dual Annealing 
    Output:
        CI - The 100(1-alpha)% credible interval 
    '''
    #f = lambda a, b: abs(a**3*(4-3*a) - b**3*(4-3*b) - (1-alpha))
    mid = np.average([a_min, b_max])
    if not brute:
        print("Finding CI with Dual Annealing")
        def F(X):
            a,b = X
            return abs(quad(f, a, b)[0] - (1-alpha))
        
        bounds = [[a_min, mid], [mid, b_max]] #can make better bounds, more consistent reliable result
        res = dict()
        res['da'] = da(F, bounds, maxiter = 2000, x0 = [a_min, b_max]) #Fairly insensitive to initial guesses, but sometimes gives bad result
        print(res['da'])
        CI = res['da']['x']
        print("The CI is ({0:0.6f},{1:0.6f})".format(CI[0], CI[1]))
        
        #make plots of parameter space and plot optiized value, out of curiosity
        plt.figure(figsize = (12,14))
        plt.title('Contour Plot of a, b Parameter Space', size = 24)
        plt.plot()
        x1 = np.linspace(a_min,mid, 100)
        x2 = np.linspace(mid, b_max, 100)
        X,Y = np.meshgrid(x1,x2)
        Z = np.zeros([len(x1), len(x2)])
        for i in range(len(x1)):
            for j in range(len(x2)):
                Z[j][i] = F([x1[i], x2[j]])
                
        plt.contour(X,Y,Z, levels = 100, colors = 'k')
        plt.contourf(X,Y,Z, levels = 100, cmap = 'binary')
        plt.plot(CI[0], CI[1], marker = '*', color = 'red', linestyle = '')
        plt.xlabel('a',size = 18)
        plt.ylabel('b',size = 18)
        plt.colorbar()
        plt.show()
        
        fig = plt.figure(figsize = (12,14))
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Surface Plot of a, b Parameter Space',size=24)
        surf = ax.plot_surface(X,Y,Z, cmap=cm.ocean)
        bci = ax.scatter(CI[0], CI[1], res['da']['fun'], color = 'r')
        ax.set_xlabel('a',size=18)
        ax.set_ylabel('b',size=18)
        ax.set_zlabel('F(a,b)',size=18)
        fig.colorbar(surf)
        plt.show()
        
        return CI
    else:
        print("Finding CI with brute force")
        a = np.arange(a_min, mid+step, step)
        b = np.arange(mid, b_max+step, step)
        
        bounds = ((x,y) for x in a for y in b) #Cartesian product of a and b; using generator func to list not explicitly created
        bounds = list(bounds) #need to work with list anyways, so convert it
    
        val = [] #hold values of integrate 
        count = 0
        n  =len(bounds)
        for el in bounds:
            count +=1
            if not count%5000:
                print("Optimizing...{0}/{1}".format(count, n))
            
            aa,bb = el
            integral,err = quad(f, aa, bb)
            if np.abs(integral - (1-alpha)) < tol:
                val.append((aa,bb))
        if not len(val):
            print("Error, could not find CI. Returning None")
            return None
        #print(val)
        m = min([(np.diff(el[1]), el[0]) for el in enumerate(val)]) #get minimum interval width and its index
        CI = val[m[1]]
        print("The CI is ({0:0.6f},{1:0.6f})".format(CI[0], CI[1]))
        
        plt.figure(figsize = (12,14))
        plt.title('Contour Plot of a, b Parameter Space',size = 24)
        plt.plot()
        x1 = np.linspace(a_min,mid, 100)
        x2 = np.linspace(mid, b_max, 100)
        X,Y = np.meshgrid(x1,x2)
        Z = np.zeros([len(x1), len(x2)])
        for i in range(len(x1)):
            for j in range(len(x2)):
                Z[j][i] = F([x1[i], x2[j]])
        plt.contour(X,Y,Z, levels = 100, colors = 'k')
        plt.contourf(X,Y,Z, levels = 100, cmap = 'binary')
        plt.plot(CI[0], CI[1], marker = '*', color = 'red', linestyle = '')
        plt.xlabel('a', size = 18)
        plt.ylabel('b', size = 18)
        plt.colorbar()
        plt.show()
        
        fig = plt.figure(figsize = (12,14))
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Surface Plot of a, b Parameter Space',size=24)
        surf = ax.plot_surface(X,Y,Z, cmap=cm.ocean)
        bci = ax.scatter(CI[0], CI[1], res['da']['fun'], color = 'r')
        ax.set_xlabel('a',size=18)
        ax.set_ylabel('b',size=18)
        ax.set_zlabel('F(a,b)',size=18)
        fig.colorbar(surf)
        plt.show()
    
        return CI

N = 3 #total coin tosses
H = 2 #total number of heads

P_theta = 1 #uninformative prior 
P_Dtheta = lambda theta: comb(N,H)* theta **H * (1-theta)**(N-H) #likelihood, from binomial distribution 
P_D = 0.25 #evidence, integrate from 0 to 1 P_theta * P_Dtheta dtheta 

P_thetaD = lambda theta: P_theta / P_D * P_Dtheta(theta) #posterior 

theta_val = np.linspace(0,1,1000) #generate analytical solution
P_thetaD_arr = P_thetaD(theta_val)

N_E = 10000 #number of trials for numerical solution

dtheta = np.linspace(0,1,21) #Theta values to consider 
n_B = np.full(len(dtheta), np.nan)
for i in range(len(dtheta)):
    x = np.random.binomial(N, dtheta[i], size = N_E)
    n_B[i] = np.sum(x==H) #count how often we get H heads in the above data
    
p_B = n_B/N_E / P_D * P_theta #n_B/N_E is P(D|theta), need to 'normalize' it to match above PDF

ci_low,ci_upp = CI_est(P_thetaD, 0, 1)
print("The width of the CI is {0:0.6f}".format(ci_upp-ci_low))
print("The CI covers {0:0.6f}% of the area".format(quad(P_thetaD, ci_low, ci_upp)[0]))


plt.figure(figsize = (12,14))
plt.title('Posterior Distribution, 95% CI = ({0:0.3}, {1:0.3})'.format(ci_low, ci_upp), size = 24)
plt.plot(theta_val, P_thetaD_arr, color = 'r', marker = 'o', linestyle = '', label = 'Analytical')
plt.bar(dtheta, p_B, dtheta[1]-dtheta[0],color = 'b', linestyle = '-', label = 'Numerical', fill = None, edgecolor = 'b')
plt.xlabel('$\\theta$', size = 18)
plt.ylabel('$P(\\theta | D)$', size = 18)
plt.axvline(ci_low, color = 'k', linestyle = '--', label = '95% BCI')
plt.axvline(ci_upp, color = 'k', linestyle = '--')
plt.legend()
plt.grid()
plt.show()
