#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:03:55 2018

@author: yangzhuronghuang
"""

import numpy as np
import time

def matrix_factorization(R,P,Q,K,bu,bi, steps=5000, alpha=0.0002, beta=0.02):
    P = P.T
    total = 0
    num = 0
    for i in range(R.shape[0]):   #user
        for j in range(R.shape[1]):  #item
            if R[i][j]>0:
                total = total+R[i][j]
                num = num+1
    mu = total/num
    for step in range(steps):
        #update every element using gradient descent
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i][j]>0:
                    eij = R[i][j]-np.dot(P[i,:],Q[:,j])-mu-bi[j]-bu[i]
                    for k in range(K):
                        P[i][k] = P[i][k]+alpha*(eij*Q[k][j]-beta*P[i][k])
                        Q[k][j] = Q[k][j]+alpha*(eij*P[i][k]-beta*Q[k][j])
                        bi[j] = bi[j]+alpha*(eij-beta*bi[j])
                        bu[i] = bu[i]+alpha*(eij-beta*bu[i])
        #calculate error at every step. If error is less than a threshold, stop
        e = 0
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i][j]>0:
                    e= e+pow(R[i][j]-np.dot(P[i,:],Q[:,j])-mu-bi[j]-bu[i],2)+beta/2*(pow(bi[j],2)+pow(bu[i],2))
                    for k in range(K):
                        e = e+beta/2*(pow(P[i][k],2)+pow(Q[k][j],2))
        if e < 0.001:
            break
    return P.T, Q, bu, bi

R = np.array([[5,3,0,1],[4,0,0,1],[1,1,0,5],[1,0,0,4],[0,1,5,4]])
u = R.shape[0]
i = R.shape[1]
K = 4
P = np.random.rand(K,u)
Q = np.random.rand(K,i)
bu = np.random.rand(u)
bi = np.random.rand(i)

t1 =time.time()
nP, nQ,ru,ri = matrix_factorization(R, P, Q, K,bu,bi)
t2 = time.time()
t = t2-t1
print('it costs {} seconds'.format(t))
nR = np.dot(nP.T,nQ)