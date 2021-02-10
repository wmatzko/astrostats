# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:01:03 2021

@author: William
"""
import numpy as np

ntrial = 100000 #number of simulations want to run, should be at least 10000

ca,cb,cc,cd = 0,0,0,0 #initialize counters for parts a - d

for i in range(ntrial):
    x = [40,40,40,40, 60, 60, 60, 60, 60, 75, 75, 75, 75,75, 75] #make list of all bulbs. Need at start because of x.remove later
    el = np.random.choice(x, 3, replace=False) #choose three from list at random
    #e75 = len([i for i in el if i == 75]) #are there two 75's in the choice?
    e75 = len(el[np.where(el==75)]) #different way 
    if e75 == 2:
        ca += 1 #if so, add to the count
    es = np.all(el[0] == el) #are all elements the same?
    if es:
        cb+=1 #if so, add to counter
    if len(np.unique(el)) == len(el): #if all elements are unique
        cc+=1 #add to counter 
    
    for j in range(len(x)): #for all the number of bulbs in x
        draw = np.random.choice(x, 1, replace=False)[0] #pick one
        x.remove(draw) #and remove from list
        if draw == 75 and j<5: #if we haven't examined 6 bulbs and we draw a 75...
            break #we're done, break loop
        elif draw == 75 and j>= 5: #if we draw 75 and we've examined at least 6 bulbs..
            cd+=1 #add to count then break
            break
        
print("The percentage for part a is {0:0.2f}%\n".format(ca/ntrial*100))
print("The percentage for part b is {0:0.2f}%\n".format(cb/ntrial*100))
print("The percentage for part c is {0:0.2f}%\n".format(cc/ntrial*100))
print("The percentage for part d is {0:0.2f}%".format(cd/ntrial*100))