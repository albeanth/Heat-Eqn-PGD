import sys,os
import os.path
import numpy as np
import math
import matplotlib.pyplot as plt

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
k = 1 # W/(m*K) -> Bulk U metal at RT
f = 1 # constant source term

X = np.array([np.linspace(0,1,61)],ndmin=2) # spatial domain -> calculated values for R for first iteration are independent of this first initial guess. tested with parabolic and linear inital shapes and same X_i+1 result was obtained.
T = np.array([np.linspace(0,0.1,151)],ndmin=2) # time domain
K = np.array([np.linspace(1,5,101)],ndmin=2) # diffusivity domain
dx = 1/61
dt = 0.1/151
dk = 5/101

## guesses for R,S,W
R = np.ones((np.shape(X)[1],1))
S = np.ones((np.shape(T)[1],1))
W = np.ones((np.shape(K)[1],1))

## Solve for R(x)
s1=0.0; s2=0.0; s3=0.0; s4=0.0; s5=0.0
for ids,dum in enumerate(S):
    if ((ids==0) or (ids==np.shape(T)[1]-1)): # boundary points
        pass
    else: # interior points
        s1 += dt/2 * (math.pow(S[ids],2)+math.pow(S[ids-1],2))
        s2 += 1/4 * (S[ids]*(S[ids+1]-S[ids]) + S[ids-1]*(S[ids]-S[ids-1]))
        s3 += dt/2 * (S[ids]+S[ids-1])
for Elem in T:
    for cnt,pair in enumerate(zip(Elem,S)):
        if ((cnt==0) or (cnt==np.shape(T)[1]-1)): #boundary points
            pass
        else: #interior points
            s4 += 1/4 * (S[cnt]*(Elem[cnt+1]-Elem[cnt]) + S[cnt-1]*(Elem[cnt]-Elem[cnt-1]))
            s5 += dt/2 * (S[cnt]*Elem[cnt] + S[cnt-1]*Elem[cnt-1])

w1=0.0; w2=0.0; w3=0.0; w4=0.0; w5=0.0
for idw,dum in enumerate(W):
    if ((idw==0) or (idw==np.shape(K)[1]-1)):
        pass
    else:
        w1 += dk/2 * (math.pow(W[idw],2)+math.pow(W[idw-1],2))
        w2 += k * dk/2 * (math.pow(W[idw],2)+math.pow(W[idw-1],2))
        w3 += dk/2 * (W[idw]+W[idw-1])
for Elem in K:
    for cnt,pair in enumerate(zip(Elem,W)):
        if ((cnt==0) or (cnt==np.shape(K)[1]-1)): #boundary points
            pass
        else:
            w4 += dk/2 * (W[cnt]*Elem[cnt]+W[cnt-1]*Elem[cnt-1])
            w5 += k * dk/2 * (W[cnt]*Elem[cnt]+W[cnt-2]*Elem[cnt-2])

v1 = -(s1*w2/(2*dx))
v2 = (s1*w2)/dx + s2*w1
v3 = v1
b = 0.0
for Elem in X:
    for cnt,idx in enumerate(Elem):
        if ((cnt==0) or (cnt==np.shape(X)[1]-1)): #boundary points
            pass
        else:
            b += -Elem[cnt-1]*(s5*w5)/(2*dx) + Elem[cnt]*((s5*w5)/(dx)+s4*w4) - Elem[cnt+1]*(s5*w5)/(2*dx)
b += f*s3*w3

R_Lmatrix = v1*np.eye(np.shape(X)[1], k=-1) + v2*np.eye(np.shape(X)[1], k=0) + v3*np.eye(np.shape(X)[1], k=1)
bVec = np.array((b*np.ones(np.shape(X)[1])))
bVec[0] = bVec[0]-v1
bVec[-1] = bVec[-1]-v3
# print(R_Lmatrix)
# sys.exit()
R = np.dot(np.linalg.inv(R_Lmatrix),bVec)
X = np.vstack((X,R))

plt.figure(1)
for cnt,elem in enumerate(X):
    plt.plot(np.linspace(0,1,61), elem/max(elem), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')
plt.show()


# ## Append X, T, and K with new iterate
# X = np.vstack((X,R))
# T = np.vstack((T,S))
# K = np.vstack((K,W))
