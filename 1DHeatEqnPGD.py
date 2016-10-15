import sys,os
import os.path
import numpy as np
import math

try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    Yellow=Fore.YELLOW; Red=Fore.RED; Green=Fore.GREEN; Cyan=Fore.CYAN; Magenta=Fore.MAGENTA
    StyDim=Style.DIM
except ImportError:
    print('\nYou should get colorama. It\'s pretty sweet.\n')
    Yellow=''; Red=''; Green=''; Cyan=''; Magenta = ''
    StyDim='';

## Initializing Info
k = 28 # W/(m*K) -> Bulk U metal at RT
f = 1 # constant source term

X = np.array([np.linspace(0,1,61)],ndmin=2) # spatial domain
dx = 1/61
T = np.array([np.linspace(0,0.1,151)],ndmin=2) # time domain
dt = 0.1/151
K = np.array([np.linspace(1,5,101)],ndmin=2) # diffusivity domain
dk = 5/101

## guesses for R,S,W
R = np.ones((np.shape(X)[1],1))
S = np.ones((np.shape(T)[1],1))
W = np.ones((np.shape(K)[1],1))

## Solve for R(x)
s1=0; s_2=0; s_3=0; s_4=0
for ids in S:
    s1 += dt/2 * (S[ids]^2+S[ids-1]^2)
    s2 += 1/4 * (S[ids]*(S[ids+1]-S[ids]) + S[ids-1]*(S[ids]-S[ids-1]))
    s3 += dt/2 * (S[ids]+S[ids-1])
for Elem in T:
    for idt,ids in zip(Elem,S):
        s4 = 1/4 * (S[idS]*(Elem[idt+1]-Elem[idt]) + S[idS-1]*(Elem[idt]-Elem[idt-1]))
        s5 = dt/2 * (S[ids]*Elem[idt] + S[ids-1]*Elem[idt-1])

w1=0; w2=0; w3=0; w4=0 w5=0
for idw in W:
    w1 += dk/2 * (W[idw]^2+W[idw-1]^2)
    w2 += k * dk/2 * (W[idw]^2+W[idw-1]^2)
    w3 += dk/2 * (W[idw]+W[idw-1])
for Elem in K:
    for idk,idw in zip(Elem,W):
        w4 += dk/2 * (W[idw]*Elem[idk]+W[idw-1]*Elem[idk-1])
        w5 += k * dk/2 * (W[idw]*Elem[idk]+W[idw-2]*Elem[idk-2])




# ## Append X, T, and K with new iterate
# X = np.vstack((X,R))
# T = np.vstack((T,S))
# K = np.vstack((K,W))
