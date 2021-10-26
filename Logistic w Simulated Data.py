#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:43:49 2021

@author: anjolaoluwapopoola
"""

# =============================================================================
# Importing necessary packages
# =============================================================================
import numpy as np
import copy
import matplotlib.pyplot as plt



# =============================================================================
# Logistic Function
# =============================================================================
def sigmoid(a):
    return 1/(1 + np.exp(-a))


# =============================================================================
# Creating dataset
# =============================================================================
np.random.seed(2020)
mu = 0
sigma = 1
X = np.random.normal(mu,sigma,size=(50,4)) #Can make low rank also by using sklearn 
vec_1 = np.ones([50,1])
X = np.concatenate((vec_1,X),axis = 1)
m = X.shape[0]
n = X.shape[1]


# =============================================================================
# Creating a response y from already determined weights y = wTx
# =============================================================================
beta = np.array([-1,2,1,4,-1])
t = X @ beta
almost_y = sigmoid(t)
y = np.zeros((m,1))
for i in range(m):
    y[i] = np.random.binomial(1,almost_y[i])

m = X.shape[0]
n = X.shape[1]



# =============================================================================
# Initializing the weights using Ordinary Least Squares
# =============================================================================
xtx= np.dot(X.T,X)
xtx_inv = np.linalg.pinv(xtx)
xty = np.dot(X.T,y)


print(xtx)
np.linalg.svd(xtx)


# =============================================================================
# Initializations
# =============================================================================
theta = np.dot(xtx_inv,xty).reshape(n,1) #weightsß
k = 0
e = 1
e_k = [1]

w_old = copy.deepcopy(theta)
w_new = copy.deepcopy(theta)


# =============================================================================
# Re-iterative Newton-Raphson Method
# =============================================================================
while e > 0.0001:
   
    k = k+1 
    
    w_old = copy.deepcopy(w_new)
    
    a = np.dot(X,theta)
    p = sigmoid(a)
    pm = np.multiply(p ,(1-p))
    W= np.diagflat(pm)
    y = y.reshape(m,1)

    inv_W = np.linalg.pinv(W)
    Z = np.dot(X,w_old) + np.dot(inv_W,(y-p))
    xtw = np.dot(X.T,W)
    xtwx = np.dot(xtw,X)
    inv_xtwx = np.linalg.pinv(xtwx)
    
    almost_w_new = np.dot(inv_xtwx,xtw)
    theta = np.dot(almost_w_new,Z)
    
    w_new = copy.deepcopy(theta)
    e = np.linalg.norm(w_new - w_old)
    e_k.append(e)
