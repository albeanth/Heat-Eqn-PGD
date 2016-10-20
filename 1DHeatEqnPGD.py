import sys,os
import os.path
import numpy as np
# from scipy import linalg
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
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
Xsize = 61
Tsize = 151
Ksize = 101

k = np.linspace(1,5,Ksize) # W/(m*K)
f = 1 # constant source term

X = np.array([np.zeros(Xsize)],ndmin=2) # spatial domain -> calculated values for R for first iteration are independent of this first initial guess. tested with parabolic and linear inital shapes and same X_i+1 result was obtained.
T = np.array([np.zeros(Tsize)],ndmin=2) # time domain
K = np.array([np.zeros(Ksize)],ndmin=2) # diffusivity domain
dx = (1-0)/Xsize
dt = (0.1-0)/Tsize
dk = (5-1)/Ksize

## guesses for R,S,W
R_old = np.ones(np.shape(X)[1])
R_old[0] = 0; R_old[-1] = 0

S_old = np.ones(np.shape(T)[1])
S_old[0] = 0
S_new = np.ones(np.shape(T)[1])
S_new[0] = 0

W_old = np.ones(np.shape(K)[1])
W_new = np.ones(np.shape(K)[1])

## convergence criteria
eps = 1E-4 # num of enrichment steps - 2-norm
nu = 1E-4 # within enrichemnt - inf-norm

NumEnr=0 # number of enrichment steps

enrCnt=0 # number of steps for an enrichment step to converge
nu_check=1
while nu_check >= nu:
    #######################
    ##   Solve for R(x)  ##
    #######################
    # print('\nSolving for R(x)')
    s1,s2,s3,s4,s5 = basis.updateS(S_old,T,dt)
    # print('updatedS -> '+str(s1)+', '+str(s2)+', '+str(s3)+', '+str(s4)+', '+str(s5))
    w1,w2,w3,w4,w5 = basis.updateW(W_old,K,dk,k)
    # print('updatedW -> '+str(w1)+', '+str(w2)+', '+str(w3)+', '+str(w4)+', '+str(w5))
    v1 = -(s1*w2/(math.pow(dx,2)))
    v2 = s2*w1 + (2*s1*w2)/math.pow(dx,2)
    v3 = v1
    b = 0.0
    for ide,Elem in enumerate(X):
        # print('R(x) '+str(Elem[0]),', '+str(Elem[-1]))
        for cnt,idx in enumerate(Elem):
            if ((cnt==0) or (cnt==Xsize-1)): #boundary points
                b += 0.0
            else:
                b += -Elem[cnt]*s4[ide]*w4[ide] + ((Elem[cnt+1]-2*Elem[cnt]+Elem[cnt-1])/math.pow(dx,2))*s5[ide]*w5[ide]
    b += f*s3*w3

    R_Lmatrix = v1*np.eye(Xsize, k=-1) + v2*np.eye(Xsize, k=0) + v3*np.eye(Xsize, k=1)
    R_Lmatrix[0,:] = [0.0]
    R_Lmatrix[0,0] = 1.0
    R_Lmatrix[-1,:] = [0.0]
    R_Lmatrix[-1,-1] = 1.0
    bVec = np.array((b*np.ones(Xsize)))
    bVec[0] = 0.0
    bVec[-1] = 0.0
    # bVec[0] = bVec[0]-v1
    # bVec[-1] = bVec[-1]-v3

    R_new = np.dot(np.linalg.inv(R_Lmatrix),bVec)
    # R_new = linalg.solve_triangular(R_Lmatrix,bVec)

    #######################
    ##   Solve for S(t)  ##
    #######################
    # print('Solving for S(t)')
    r1,r2,r3,r4,r5 = basis.updateR(R_new,X,dx)
    # print('updatedR -> '+str(r1)+', '+str(r2)+', '+str(r3)+', '+str(r4)+', '+str(r5))
    # All 'w' integrals already solved for in R(x) solve. Don't need to resolve them.

    u1 = -(r1*w1)/(2*dt)
    u2 = -w2*r2
    u3 = -u1
    tmp = 0.0
    for ide,Elem in enumerate(T):
        # print('S(t) '+str(Elem[0]),', '+str(Elem[-1]))
        for cnt,idx in enumerate(Elem):
            if ((cnt==0) or (cnt==Tsize-1)): #boundary points
                tmp += 0.0
            else:
                tmp += -((Elem[cnt+1]-Elem[cnt])/dt)*r5[ide]*w4[ide] + Elem[cnt]*w5[ide]*r4[ide]
    for idy in np.arange(0,Tsize-1):
        S_new[idy+1] = (tmp + f*r3*w3 + S_new[idy]*w1*r1/dt)/(w1*r1/dt - w2*r2)

    #######################
    ##   Solve for W(k)  ##
    #######################
    # print('Solving for W(k)')
    r1,r2,r3,r4,r5 = basis.updateR(R_new,X,dx)
    # print('updatedR -> '+str(r1)+', '+str(r2)+', '+str(r3)+', '+str(r4)+', '+str(r5))
    s1,s2,s3,s4,s5 = basis.updateS(S_new,T,dt)
    # print('updatedS -> '+str(s1)+', '+str(s2)+', '+str(s3)+', '+str(s4)+', '+str(s5))

    for idk,param in enumerate(k): # each element in conductivity range
        tmp = 0.0
        for ide,Elem in enumerate(K): # each enrichment step
            for value in Elem:
                tmp += -(value*r5[ide]*s4[ide] - param*value*r4[ide]*s5[ide])
        tmp += f*r3*s3
        tmp = tmp/(r1*s2-param*s1*r2)
        W_new[idk] = tmp


    #######################################
    ## Check for enrichment convergence  ##
    #######################################

    # new = np.outer(np.outer(R_new,S_new),W_new)
    # new = np.tensordot(np.tensordot(R_new,S_new,axes=0),W_new,axes=0)
    # print(np.shape(new))
    # old = np.tensordot(np.tensordot(R_old,S_old,axes=0),W_old,axes=0)
    # print(np.shape(old))
    # nu_check = np.linalg.norm((new-old),np.inf)/np.linalg.norm(old,np.inf)
    # nu_check = abs(np.amax(new-old))/abs(np.amax(old))
    # print(abs(np.amax(new-old)))
    # print(abs(np.amax(old)))

    nu_check = 0.0
    for idr in np.arange(0,len(R_new)):
        for ids in np.arange(0,len(S_new)):
            for idw in np.arange(0,len(W_new)):
                tmp = abs((R_new[idr]*S_new[ids]*W_new[idw] - R_old[idr]*S_old[ids]*W_old[idw]))
                if tmp > nu_check:
                    nu_check = tmp
                else:
                    pass

    print(Cyan+str(nu_check))
    # sys.exit()
    if nu_check >= nu:
        R_old = R_new
        S_old = S_new
        W_old = W_new
        enrCnt += 1
        # if enrCnt == 1:
        #     print(Red+'maximum iteration count reached.')
        #     X = np.vstack((X,R_new))
        #     T = np.vstack((T,S_new))
        #     K = np.vstack((K,W_new))
        #     break
    else:
        NumEnr += 1 # counter for number of steps for enrichment step convergence
        X = np.vstack((X,R_new))
        T = np.vstack((T,S_new))
        K = np.vstack((K,W_new))
        print(Green+'\nEnrichment Step '+str(NumEnr)+' completed in '+str(enrCnt)+' steps.\n')
        break

plt.figure(1)
for cnt,elem in enumerate(X):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(0,1,Xsize), elem, label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(2)
for cnt,elem in enumerate(T):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(0,0.1,Tsize), elem, label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(3)
for cnt,elem in enumerate(K):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(1,5,Ksize), elem, label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')
plt.show()
