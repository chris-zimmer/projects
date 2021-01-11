# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:35:01 2020

@author: zimme
"""

import numpy as np
from random import sample, choices
from math import floor

def MetroHast(n, T):
    l = [0,1]*(int(n/2))
    x = choices(l, k=n)
    u = []
    #Generate the neighbors of x:
    a = [[0]*len(x)]*len(x)
    for i in range(0,len(x)):
        if x[i]==0:
            a[i] = sample(x, len(x))
            a[i][i] = 1
        else:
            a[i] = sample(x, len(x))
            a[i][i] = 0

    #############################
    for t in range(0,T):
        e = int(n-len(a))
        q = [[x]]*e
        p = a+q
        v = sample(p, 1)
        if binaryentropy(v, n) >= binaryentropy(x, n):
            u.append(v)
        else:
            y = floor((binaryentropy(v, n)/binaryentropy(x, n))*100)
            b = [[v]*int(y)]
            c = [[x]*(100-int(y))]
            d = b+c
            z = sample(d,1)
            u.append(z)
    return u
            

def binaryentropy(v,n):
    w = (np.sum(v))/n
    H = w*np.log2(1/w) + (1-w)*np.log2(1/(1-w))
    return H


u = MetroHast(100, 10000)
uu = MetroHast(100, 100)


