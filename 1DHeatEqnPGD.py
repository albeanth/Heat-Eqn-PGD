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

#######################
##   Solve for R(x)  ##
#######################
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
            b += -Elem[cnt-1]*(s5*w5)/(2*dx) + Elem[cnt]*(((s5*w5)/dx)+s4*w4) - Elem[cnt+1]*(s5*w5)/(2*dx)
b += f*s3*w3

R_Lmatrix = v1*np.eye(np.shape(X)[1], k=-1) + v2*np.eye(np.shape(X)[1], k=0) + v3*np.eye(np.shape(X)[1], k=1)
bVec = np.array((b*np.ones(np.shape(X)[1])))
bVec[0] = bVec[0]-v1
bVec[-1] = bVec[-1]-v3
R = np.dot(np.linalg.inv(R_Lmatrix),bVec)

#######################
##   Solve for S(t)  ##
#######################
r1=0; r2=0; r3=0; r4=0; r5=0
for idr,dum in enumerate(R):
    if ((idr==0) or (idr==np.shape(X)[1]-1)): # boundary points
        pass
    else: # interior points
        r1 += dx/2 * (math.pow(R[idr],2)+math.pow(R[idr-1],2))
        if idr == 1:
            r2+= math.pow(dx,3)/8 * (R[idr]*(R[idr]-2*R[idr+1]+R[idr]) + R[idr-1]*(R[idr-1]-2*R[idr]+R[idr+1]))
        else:
            r2 += 1/4 * (R[idr]*(R[idr+1]-2*R[idr]+R[idr-1]) + R[idr-1]*(R[idr]-2*R[idr-1]+R[idr-2]))
        r3 += dx/2 * (R[idr]+R[idr-1])
for Elem in X:
    for cnt,pair in enumerate(zip(Elem,R)):
        if ((cnt==0) or (cnt==np.shape(X)[1]-1)): #boundary points
            pass
        else: #interior points
            if cnt ==1:
                r4 += math.pow(dx,3)/8 * (R[cnt]*(Elem[cnt]-2*Elem[cnt+1]+Elem[cnt]) + R[cnt-1]*(Elem[cnt-1]-2*Elem[cnt]+Elem[cnt+1]))
            else:
                r4 += 1/4 * (R[cnt]*(R[cnt+1]-2*R[cnt]+R[cnt-1]) + R[cnt-1]*(R[cnt]-2*R[cnt-1]+R[cnt-2]))
            r5 += dx/2 * (R[cnt]*Elem[cnt]+R[cnt-1]*Elem[cnt-1])

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
S = np.dot(np.linalg.inv(S_Lmatrix),yVec)

#######################
##   Solve for W(k)  ##
#######################
r1=0; r2=0; r3=0; r4=0; r5=0
for idr,dum in enumerate(R):
    if ((idr==0) or (idr==np.shape(X)[1]-1)): # boundary points
        pass
    else: # interior points
        r1 += dx/2 * (math.pow(R[idr],2)+math.pow(R[idr-1],2))
        if idr == 1:
            r2+= math.pow(dx,3)/8 * (R[idr]*(R[idr]-2*R[idr+1]+R[idr]) + R[idr-1]*(R[idr-1]-2*R[idr]+R[idr+1]))
        else:
            r2 += 1/4 * (R[idr]*(R[idr+1]-2*R[idr]+R[idr-1]) + R[idr-1]*(R[idr]-2*R[idr-1]+R[idr-2]))
        r3 += dx/2 * (R[idr]+R[idr-1])
for Elem in X:
    for cnt,pair in enumerate(zip(Elem,R)):
        if ((cnt==0) or (cnt==np.shape(X)[1]-1)): #boundary points
            pass
        else: #interior points
            if cnt ==1:
                r4 += math.pow(dx,3)/8 * (R[cnt]*(Elem[cnt]-2*Elem[cnt+1]+Elem[cnt]) + R[cnt-1]*(Elem[cnt-1]-2*Elem[cnt]+Elem[cnt+1]))
            else:
                r4 += 1/4 * (R[cnt]*(R[cnt+1]-2*R[cnt]+R[cnt-1]) + R[cnt-1]*(R[cnt]-2*R[cnt-1]+R[cnt-2]))
            r5 += dx/2 * (R[cnt]*Elem[cnt]+R[cnt-1]*Elem[cnt-1])

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
W = np.dot(np.linalg.inv(W_Lmatrix),zVec)


X = np.vstack((X,R))
T = np.vstack((T,S))
K = np.vstack((K,W))
plt.figure(1)
for cnt,elem in enumerate(X):
    plt.plot(np.linspace(0,1,61), elem/max(elem), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(2)
for cnt,elem in enumerate(T):
    plt.plot(np.linspace(0,1,151), elem/max(elem), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(3)
for cnt,elem in enumerate(K):
    plt.plot(np.linspace(0,1,101), elem/max(elem), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')
plt.show()
#
#
# ## Check for enrichment convergence
# tmp = np.outer(R,S)
# tmp2 = np.outer(X,T)
# ### epscheck should be reworked into the 2-norm ###
# epscheck = max(np.outer(tmp,W))/max(np.outer(tmp,K)) # is inf-norm
# if epscheck >= eps:
#     # Append X, T, and K with new iterate
    # X = np.vstack((X,R))
    # T = np.vstack((T,S))
    # K = np.vstack((K,W))
# else:
#     # enrichement process is done!
#     break
