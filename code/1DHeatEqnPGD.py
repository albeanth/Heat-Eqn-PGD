import sys,os
import os.path
import numpy as np
import math
import matplotlib.pyplot as plt
import basis
import linecache
from tqdm import tqdm

import HeatPGD_Error
import HeatPGDPostCompress
import exact as ref

try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    Yellow=Fore.YELLOW; Red=Fore.RED; Green=Fore.GREEN; Cyan=Fore.CYAN; Magenta=Fore.MAGENTA
    StyDim=Style.DIM
except ImportError:
    print('\nYou should get colorama. It\'s pretty sweet.\n')
    Yellow=''; Red=''; Green=''; Cyan=''; Magenta = ''
    StyDim='';

## General Functions
def normSqRt(Vec,size,stp):
    tmp = 0.0
    for idx in np.arange(0,size-1):
        tmp += stp/2*(Vec[idx+1]**2 + Vec[idx]**2)
    return(math.sqrt(tmp))

def norm(Vec,size,stp):
    tmp = 0.0
    for idx in np.arange(0,size-1):
        tmp += stp/2*(Vec[idx+1] + Vec[idx])
    return(tmp)

## Initializing Info
Xsize = 61
Tsize = 151
Ksize = 101

Xrange = 1-0
Trange = 0.1-0
Krange = 5-1
dx = (Xrange)/(Xsize-1)
dt = (Trange)/(Tsize-1)
dk = (Krange)/(Ksize-1)

space = np.linspace(0,1,Xsize)
time = np.linspace(0,0.1,Tsize)

cond = np.linspace(1,5,Ksize) # W/(m*K)
f = 1.0 # constant source term

# ExactSoln = ref.exact(space,time,cond, Xsize,Tsize,Ksize)

Max_fp_iter = 50
MaxEnr = 25
###############################################################################

X = np.array([np.ones(Xsize)],ndmin=2) # spatial domain -> calculated values for R for first iteration are independent of this first initial guess. tested with parabolic and linear inital shapes and same X_i+1 result was obtained.
T = np.array([np.ones(Tsize)],ndmin=2) # time domain
K = np.array([np.ones(Ksize)],ndmin=2) # diffusivity domain

## convergence criteria
eps = 1E-8 # num of enrichment steps - 2-norm
nu = 1E-8 # within enrichemnt - inf-norm

NumEnr=0 # number of enrichment steps
eps_check = 1.0
while eps_check > eps:
    NumEnr += 1 # counter for number of steps for enrichment step convergence

    R_old = np.ones(Xsize)
    R_old[0] = 0.0; R_old[-1] = 0.0

    S_old = np.ones(Tsize)
    S_old[0] = 0.0

    W_old = np.ones(Ksize)

    enrCnt=0 # number of steps for an enrichment step to converge
    nu_check=1
    s1,s2,s3,s4,s5 = basis.updateS(S_old,T,dt,NumEnr)

    while nu_check >= nu:

        #######################
        ##   Solve for R(x)  ##
        #######################
        w1,w2,w3,w4,w5 = basis.updateW(W_old,K,dk,cond,NumEnr)

        v1 = -(s1*w2/(math.pow(dx,2)))
        v2 = s2*w1 + (2*s1*w2)/math.pow(dx,2)
        v3 = v1

        b = np.zeros(Xsize)
        for idr in np.arange(0,Xsize):
            if ((idr==0) or (idr==Xsize-1)): #boundary points
                b[idr] = 0.0
            else:
                for ide,Elem in enumerate(X):
                    b[idr] += -Elem[idr]*s4[ide]*w4[ide] + ((Elem[idr+1]-2*Elem[idr]+Elem[idr-1])/math.pow(dx,2))*s5[ide]*w5[ide]
                b[idr] += f*s3*w3

        R_Lmatrix = v1*np.eye(Xsize, k=-1) + v2*np.eye(Xsize, k=0) + v3*np.eye(Xsize, k=1)
        R_Lmatrix[0,:] = [0.0]
        R_Lmatrix[0,0] = 1.0
        R_Lmatrix[-1,:] = [0.0]
        R_Lmatrix[-1,-1] = 1.0

        R_new = np.linalg.solve(R_Lmatrix,b)
        R_new[0] = 0.0

        #######################
        ##   Solve for S(t)  ##
        #######################
        r1,r2,r3,r4,r5 = basis.updateR(R_new,X,dx,NumEnr)

        S_new = np.ones(Tsize)
        S_new[0] = 0.0
        for idy in np.arange(0,Tsize-1):
            tmp = 0.0
            for ide,Elem in enumerate(T):
                tmp += r5[ide]*w4[ide]*(Elem[idy+1]-Elem[idy])/dt - Elem[idy+1]*w5[ide]*r4[ide]
            S_new[idy+1] = (S_new[idy]*w1*r1/dt - tmp + w3*r3*f)/(w1*r1/dt - w2*r2)

        #######################
        ##   Solve for W(k)  ##
        #######################
        # All 'r' integrals already solved for in R(x) solve. Don't need to resolve them.
        s1,s2,s3,s4,s5 = basis.updateS(S_new,T,dt,NumEnr)

        W_new = np.ones(Ksize)
        for idk,param in enumerate(cond): # each element in conductivity range
            tmp = f*r3*s3
            for ide,Elem in enumerate(K): # each enrichment step
                tmp += -Elem[idk]*r5[ide]*s4[ide] + param*Elem[idk]*r4[ide]*s5[ide]
            W_new[idk] = tmp/(r1*s2-param*s1*r2)

        #######################################
        ## Check for enrichment convergence  ##
        #######################################

        ## actual convergence
        intR_new = 0.0; intR_old = 0.0; intR_both = 0.0
        for idr in np.arange(0,Xsize-1):
            intR_new += dx/2*(R_new[idr+1]**2 + R_new[idr]**2)
            intR_old += dx/2*(R_old[idr+1]**2 + R_old[idr]**2)
            intR_both += dx/2*(R_new[idr+1]*R_old[idr+1]+R_new[idr]*R_old[idr])

        intS_new=0.0; intS_old=0.0; intS_both = 0.0
        for ids in np.arange(0,Tsize-1):
            intS_new += dt/2*(S_new[ids+1]**2 + S_new[ids]**2)
            intS_old += dt/2*(S_old[ids+1]**2 + S_old[ids]**2)
            intS_both += dt/2*(S_new[ids+1]*S_old[ids+1]+S_new[ids]*S_old[ids])

        intW_new=0.0; intW_old=0.0; intW_both = 0.0
        for idw in np.arange(0,Ksize-1):
            intW_new += dk/2*(W_new[idw+1]**2 + W_new[idw]**2)
            intW_old += dk/2*(W_old[idw+1]**2 + W_old[idw]**2)
            intW_both += dk/2*(W_new[idw+1]*W_old[idw+1]+W_new[idw]*W_old[idw])

        new = intR_new*intS_new*intW_new
        old = intR_old*intS_old*intW_old
        nu_check = math.sqrt( (new-old)**2 ) / math.sqrt(old**2) #my way - Eq 2.8

        # print(Cyan+str(nu_check))
        enrCnt += 1
        if nu_check >= nu:
            if enrCnt > Max_fp_iter-1:
                print(Yellow+'\nThe maximum number of iterations reached. Passing onto next enrichment step.')
                print('Enrichment Step '+str(NumEnr)+' completed in '+str(enrCnt)+' steps.');
                break
            R_old = R_new
            S_old = S_new
            W_old = W_new
        else:
            print('\nEnrichment Step '+str(NumEnr)+' completed in '+str(enrCnt)+' steps.')
            break

    ## Enrichment step has converged, tack it onto to X, T, and K and check the overal convergence
    R_new_Norm = math.sqrt(intR_new/(Xrange**2))
    S_new_Norm = math.sqrt(intS_new/(Trange**2))
    W_new_Norm = math.sqrt(intW_new/(Krange**2))
    RSW_new = math.pow((R_new_Norm*S_new_Norm*W_new_Norm),1/3)
    # print(R_new_Norm,S_new_Norm,W_new_Norm,RSW_new)
    if NumEnr == 1:
        X[0] = RSW_new*R_new/R_new_Norm
        T[0] = RSW_new*S_new/S_new_Norm
        K[0] = RSW_new*W_new/W_new_Norm
    else:
        X = np.vstack((X,RSW_new*R_new/R_new_Norm))
        T = np.vstack((T,RSW_new*S_new/S_new_Norm))
        K = np.vstack((K,RSW_new*W_new/W_new_Norm))

    ###############################################
    ## Check for overall enrichment convergence  ##
    ###############################################
    R1 = 0.0; S1 = 0.0; W1 = 0.0
    for idr in np.arange(0,Xsize-1):
        R1 += dx/2*(X[0][idr+1]**2 + X[0][idr]**2)
    for ids in np.arange(0,Tsize-1):
        S1 += dt/2*(T[0][ids+1]**2 + T[0][ids]**2)
    for idw in np.arange(0,Ksize-1):
        W1 += dk/2*(K[0][idw+1]**2 + K[0][idw]**2)

    eps_check = math.sqrt(intR_new*intS_new*intW_new)/math.sqrt(R1*S1*W1)

    print(Cyan+'Error = {0:.12e}'.format(eps_check))

    if eps_check >= eps:
        if NumEnr == MaxEnr:
            print(Yellow+'\nThe maximum number of enrichment iterations reached. Solution error is '+str(eps_check)+'\n'); break
    else:
        print(Green+'Done. It took '+str(NumEnr)+ ' enrichment steps to converge solution.\n')
        break

###############################################################################

###############################
## Plot enrichment functions ##
###############################

Xout = np.zeros((Xsize,4))
plt.figure(1)
for cnt,elem in enumerate(X[0:4]):
    Xout[:,cnt] = elem/normSqRt(elem,Xsize,dx)
    plt.plot(space, Xout[:,cnt], label = str(cnt), linewidth = 3) #elem/norm(elem,Xsize,dx)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')
plt.xlabel('space domain')

Tout = np.zeros((Tsize,4))
plt.figure(2)
for cnt,elem in enumerate(T[0:4]):
    Tout[:,cnt] = elem/normSqRt(elem,Tsize,dt)
    plt.plot(time, Tout[:,cnt], label = str(cnt), linewidth = 3) #elem/norm(elem,Tsize,dt)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')
plt.xlabel('time domain')

Kout = np.zeros((Ksize,4))
plt.figure(3)
for cnt,elem in enumerate(K[0:4]):
    Kout[:,cnt] = elem/normSqRt(elem,Ksize,dk)
    plt.plot(cond, Kout[:,cnt], label = str(cnt), linewidth = 3) #elem/norm(elem,Ksize,dk)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')
plt.xlabel('parameter space')

#################################
## Writeout X,T,K data to file ##
#################################

fileout = open('../data/Raw_BasisFunc.txt','w')
for idx in np.arange(0,Xsize):
    fileout.write('{0:.4f}\t'.format( space[idx] ))
    for enr in np.arange(0,Xout.shape[1]):
        fileout.write('{0:.8e}\t'.format(Xout[idx][enr]))
    fileout.write('\n')

fileout.write('\n\n')
for idt in np.arange(0,Tsize):
    fileout.write('{0:.4f}\t'.format( time[idt] ))
    for enr in np.arange(0,Tout.shape[1]):
        fileout.write('{0:.8e}\t'.format(Tout[idt][enr]))
    fileout.write('\n')

fileout.write('\n\n')
for idk in np.arange(0,Ksize):
    fileout.write('{0:.4f}\t'.format( cond[idk] ))
    for enr in np.arange(0,Kout.shape[1]):
        fileout.write('{0:.8e}\t'.format(Kout[idk][enr]))
    fileout.write('\n')

fileout.close()
plt.show()
sys.exit()

######################################################################
## Complete Post-Compression to improve optimality of PGD solution  ##
######################################################################
print(Magenta+'\n\nStarting post-compression...')
Xc,Tc,Kc = HeatPGDPostCompress.PGD_PostComp(eps,nu,Max_fp_iter,MaxEnr, X,T,K, dx,dt,dk,Xsize,Tsize,Ksize)

#################################################################
## Error Calculations for PGD as a function of enrichment step ##
#################################################################
figCnt = 4
print(Magenta+'\n\nGetting error of raw and compressed PGD approximations...')
HeatPGD_Error.XD_Error(space,cond, X,T,K, Xc,Tc,Kc, Xsize,Tsize,Ksize, dx,dt,dk, ExactSoln, figCnt)
