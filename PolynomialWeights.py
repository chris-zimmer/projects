# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:39:23 2020

@author: zimme
"""
from random import sample, choices
import numpy as np
import random
import bisect
import collections

#Global parameters:
N = [0, 1, 2]
w = [1, 1, 1]
action = 0
loss_i = 0

#The agent:
def my_agent(observation, configuration):
    global N 
    global w
    global action
    global loss_i
    e = np.sqrt(np.log(3)/(observation.step+1))
    loss_i = loss_from_previous_action(observation, action)
    for i in range(0, 3):
        w[i] = w[i]*(1-e*loss_i)
    action = choice(N, w)
    return action

#Recursive functions for the algorithm:
def loss_from_previous_action(observation, action):
    global loss_i
    if action == observation.lastOpponentAction:
        loss_i = 0.5
    elif (action == 0 and observation.lastOpponentAction == 1) or (action == 1 and observation.lastOpponentAction == 2) or (action == 2 and observation.lastOpponentAction == 0):
        loss_i = 1
    else:
        loss_i = 0
    return loss_i

def choice(population, weights):
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


