import sys,os
import os.path
import numpy as np
import math
import matplotlib.pyplot as plt
import basis

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
R_old = np.ones(np.shape(X)[1])
S_old = np.ones(np.shape(T)[1])
W_old = np.ones(np.shape(K)[1])

## convergence criteria
eps = 1E-4 # num of enrichment steps
nu = 1E-4 # within enrichemnt

NumEnr=0 # number of enrichment steps

enrCnt=0 # number of steps for an enrichment step to converge
nu_check=1
while nu_check >= nu:
    #######################
    ##   Solve for R(x)  ##
    #######################

    s1,s2,s3,s4,s5 = basis.updateS(S_old,T,dt)
    w1,w2,w3,w4,w5 = basis.updateW(W_old,K,dk,k)

    v1 = -(s1*w2/(2*dx))
    v2 = (s1*w2)/dx + s2*w1
    v3 = v1
    b = 0.0
    for Elem in X:
        for cnt,idx in enumerate(Elem):
            if ((cnt==0) or (cnt==np.shape(X)[1]-1)): #boundary points
                pass
            else:
                b += -Elem[cnt-1]*(s5*w5)/(2*dx) + Elem[cnt]*((s5*w5/dx)+s4*w4) - Elem[cnt+1]*(s5*w5)/(2*dx)
    b += f*s3*w3

    R_Lmatrix = v1*np.eye(np.shape(X)[1], k=-1) + v2*np.eye(np.shape(X)[1], k=0) + v3*np.eye(np.shape(X)[1], k=1)
    bVec = np.array((b*np.ones(np.shape(X)[1])))
    bVec[0] = bVec[0]-v1
    bVec[-1] = bVec[-1]-v3

    R_new = np.dot(np.linalg.inv(R_Lmatrix),bVec)

    #######################
    ##   Solve for S(t)  ##
    #######################
    r1,r2,r3,r4,r5 = basis.updateR(R_new,X,dx)
    # All 'w' integrals already solved for in R(x) solve. Don't need to resolve them.

    u1 = -(r1*w1)/(2*dt)
    u2 = -w2*r2
    u3 = -u1
    y = 0.0
    for Elem in T:
        for cnt,idx in enumerate(Elem):
            if ((cnt==0) or (cnt==np.shape(T)[1]-1)): #boundary points
                pass
            else:
                y += -Elem[cnt-1]*r5*w4/(2*dt) - Elem[cnt]*w5*r4 + Elem[cnt+1]*r5*w4/(2*dt)
    y += f*r3*w3

    S_Lmatrix = u1*np.eye(np.shape(T)[1], k=-1) + u2*np.eye(np.shape(T)[1], k=0) + u3*np.eye(np.shape(T)[1], k=1)
    yVec = np.array((y*np.ones(np.shape(T)[1])))
    yVec[0] = yVec[0] - u1
    yVec[-1] = yVec[-1] - u3

    S_new = np.dot(np.linalg.inv(S_Lmatrix),yVec)

    #######################
    ##   Solve for W(k)  ##
    #######################
    r1,r2,r3,r4,r5 = basis.updateR(R_new,X,dx)
    s1,s2,s3,s4,s5 = basis.updateS(S_new,T,dt)

    z = 0.0
    for Elem in K:
        for cnt,idx in enumerate(Elem):
            if ((cnt==0) or (cnt==np.shape(T)[1]-1)): #boundary points
                pass
            else:
                z += -(Elem[cnt]*r5*s4 - k*Elem[cnt]*r4*s5)
    z += f*r3*s3
    z = z/(r1*s2-k*s1*r2)

    W_Lmatrix = np.eye(np.shape(K)[1], k=0)
    zVec = np.array((z*np.ones(np.shape(K)[1])))

    W_new = np.dot(np.linalg.inv(W_Lmatrix),zVec)

    #######################################
    ## Check for enrichment convergence  ##
    #######################################
    # new = np.outer(np.outer(R_new,S_new),W_new)
    new = np.tensordot(np.tensordot(R_new,S_new,axes=0),W_new,axes=0)
    # print(np.shape(new))
    old = np.tensordot(np.tensordot(R_old,S_old,axes=0),W_old,axes=0)
    # print(np.shape(old))

    # nu_check = np.linalg.norm((new-old),np.inf)/np.linalg.norm(old,np.inf)
    nu_check = np.amax(new-old)/np.amax(old)
    print(str(nu_check))
    if nu_check >= nu:
        R_old = R_new
        S_old = S_new
        W_old = W_new
        enrCnt += 1
    else:
        NumEnr += 1 # counter for number of steps for enrichment step convergence
        X = np.vstack((X,R_new))
        T = np.vstack((T,S_new))
        K = np.vstack((K,W_new))
        print(Green+'\nEnrichment Step '+str(NumEnr)+' completed in '+str(enrCnt)+' steps.\n')
        break

# ### epscheck should be reworked into the 2-norm ###
# epscheck = max(np.outer(tmp,W))/max(np.outer(tmp,K)) # is inf-norm
# if epscheck >= eps:
# else:
#     # enrichement process is done!
#     break

plt.figure(1)
for cnt,elem in enumerate(X):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(0,1,61), elem/max(elem), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(2)
for cnt,elem in enumerate(T):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(0,1,151), elem/max(elem), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(3)
for cnt,elem in enumerate(K):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(0,1,101), elem/max(elem), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')
plt.show()
