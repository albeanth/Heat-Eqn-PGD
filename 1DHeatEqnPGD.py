import sys,os
import os.path
import numpy as np
import math
import matplotlib.pyplot as plt
import basis
import linecache
from tqdm import tqdm

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

Xrange = 1-0
Trange = 0.1-0
Krange = 5-1
dx = (Xrange)/(Xsize-1)
dt = (Trange)/(Tsize-1)
dk = (Krange)/(Ksize-1)

k = np.linspace(1,5,Ksize) # W/(m*K)
f = 1 # constant source term

Max_fp_iter = 50
MaxEnr = 25
###############################################################################

X = np.array([np.ones(Xsize)],ndmin=2) # spatial domain -> calculated values for R for first iteration are independent of this first initial guess. tested with parabolic and linear inital shapes and same X_i+1 result was obtained.
T = np.array([np.ones(Tsize)],ndmin=2) # time domain
K = np.array([np.ones(Ksize)],ndmin=2) # diffusivity domain

## convergence criteria
eps = 1E-4 # num of enrichment steps - 2-norm
nu = 1E-8 # within enrichemnt - inf-norm

NumEnr=0 # number of enrichment steps
eps_check = 1.0
while eps_check > eps:
    NumEnr += 1 # counter for number of steps for enrichment step convergence
    # apparently, each iteration begins with a guess of ones.
    R_old = np.ones(Xsize)
    R_old[0] = 0; R_old[-1] = 0

    S_old = np.ones(Tsize)
    S_old[0] = 0
    S_new = np.ones(Tsize)
    S_new[0] = 0

    W_old = np.ones(Ksize)
    W_new = np.ones(Ksize)

    enrCnt=0 # number of steps for an enrichment step to converge
    nu_check=1
    s1,s2,s3,s4,s5 = basis.updateS(S_old,T,dt,NumEnr)
    # print('s integrals = {0:.8e}, {1:.8e}, {2:.8e}, '.format(s1,s2,s3)+str(s4)+' '+str(s5))
    while nu_check >= nu:
        #######################
        ##   Solve for R(x)  ##
        #######################
        w1,w2,w3,w4,w5 = basis.updateW(W_old,K,dk,k,NumEnr)
        # print('w integrals = {0:.8e}, {1:.8e}, {2:.8e}, '.format(w1,w2,w3)+str(w4)+' '+str(w5))
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

        R_new = np.dot(np.linalg.inv(R_Lmatrix),b)
        # R_new = np.linalg.solve(R_Lmatrix,b)

        #######################
        ##   Solve for S(t)  ##
        #######################
        r1,r2,r3,r4,r5 = basis.updateR(R_new,X,dx,NumEnr)
        # print('r integrals = {0:.8e}, {1:.8e}, {2:.8e}, '.format(r1,r2,r3)+str(r4)+' '+str(r5))
        # All 'w' integrals already solved for in R(x) solve. Don't need to resolve them.

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
        # print('s integrals = {0:.8e}, {1:.8e}, {2:.8e}, '.format(s1,s2,s3)+str(s4)+' '+str(s5))
        for idk,param in enumerate(k): # each element in conductivity range
            tmp = 0.0
            for ide,Elem in enumerate(K): # each enrichment step
                tmp += Elem[idk]*r5[ide]*s4[ide] - param*Elem[idk]*r4[ide]*s5[ide]
            W_new[idk] = (-tmp + f*r3*s3)/(r1*s2-param*s1*r2)

        #######################################
        ## Check for enrichment convergence  ##
        #######################################

        ## actual convergence
        intR_new = 0.0; intR_old = 0.0; intR_both = 0.0
        for idr in np.arange(0,Xsize-1):
            intR_new += dx/2*(R_new[idr+1]**2 + R_new[idr]**2)
            intR_old += dx/2*(R_old[idr+1]**2 + R_old[idr]**2)
            intR_both += dx/2*(R_new[idr+1]*R_old[idr+1]+R_new[idr]*R_old[idr])
        # print('convergence params \n',str(intR_new), str(intR_old), str(intR_both))
        # sys.exit()

        intS_new=0.0; intS_old=0.0; intS_both = 0.0
        for ids in np.arange(0,Tsize-1):
            intS_new += dt/2*(S_new[ids+1]**2 + S_new[ids]**2)
            intS_old += dt/2*(S_old[ids+1]**2 + S_old[ids]**2)
            intS_both += dt/2*(S_new[ids+1]*S_old[ids+1]+S_new[ids]*S_old[ids])
        # print(intS_new, intS_old, intS_both)
        # sys.exit()

        intW_new=0.0; intW_old=0.0; intW_both = 0.0
        for idw in np.arange(0,Ksize-1):
            intW_new += dk/2*(W_new[idw+1]**2 + W_new[idw]**2)
            intW_old += dk/2*(W_old[idw+1]**2 + W_old[idw]**2)
            intW_both += dk/2*(W_new[idw+1]*W_old[idw+1]+W_new[idw]*W_old[idw])
        # print(intW_new, intW_old, intW_both)
        # sys.exit()

        # print((intR_new*intS_new*intW_new) + (intR_old*intS_old*intW_old))
        # print(2*(intR_both*intS_both*intW_both))
        # sys.exit()
        nu_check = math.sqrt( (intR_new*intS_new*intW_new) + (intR_old*intS_old*intW_old) - 2*(intR_both*intS_both*intW_both) )

        ## palmer's interpretation
        # nu_check = 0.0
        # for idr in np.arange(0,Xsize):
        #     for ids in np.arange(0,Tsize):
        #         for idw in np.arange(0,Ksize):
        #             tmp = abs(R_new[idr]*S_new[ids]*W_new[idw] - R_old[idr]*S_old[ids]*W_old[idw])
        #             if tmp > nu_check:
        #                 nu_check = tmp
        #             else:
        #                 pass
        #
        ## tony's interpretation
        # Num = 0.0; Denom = 0.0
        # for xNew,xOld in zip(R_new,R_old):
        #     for tNew,tOld in zip(S_new,S_old):
        #         for kNew,kOld in zip(W_new,W_old):
        #             new = xNew*tNew*kNew
        #             old = xOld*tOld*kOld
        #             Num += abs(new - old)
        #             # Denom += abs(old)
        # # nu_check = Num/Denom
        # nu_check = Num

        # print(Cyan+str(nu_check))
        enrCnt += 1
        if nu_check >= nu:
            if enrCnt > Max_fp_iter-1:
                print(Red+'\nThe maximum number of iterations reached. Passing onto next enrichment step.'); break
            R_old = R_new
            S_old = S_new
            W_old = W_new
        else:
            break

    ## Enrichment step has converged, tack it onto to X, T, and K and check the overal convergence
    print('\nEnrichment Step '+str(NumEnr)+' completed in '+str(enrCnt)+' steps.')
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
    # print(Yellow+'   ....checking enrichment convergence')

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

    ## tony's interpretation
    # TotSum = 0.0; NewSum = 0.0
    # for enr in np.arange(0,len(X)):
    #     for idr in np.arange(0,Xsize):
    #         for idt in np.arange(0,Tsize):
    #             for idk in np.arange(0,Ksize):
    #                 TotSum += abs(X[enr][idr]*T[enr][idt]*K[enr][idk])
    #                 if enr == len(X)-1:
    #                     NewSum += abs(X[enr][idr]*T[enr][idt]*K[enr][idk])
    # eps_check = NewSum/TotSum

    print(Yellow+'Error = '+str(eps_check))
    if eps_check >= eps:
        if NumEnr == MaxEnr:
            print(Red+'\nThe maximum number of enrichment iterations reached. Solution error is '+str(eps_check)+'\n'); break
    else:
        print(Green+'Done. It took '+str(NumEnr)+ ' enrichment steps to converge solution.\n')
        break

###############################################################################

def normSqRt(Vec,size,stp):
    tmp = 0.0
    for idx in np.arange(0,size-1):
        tmp += stp/2*(Vec[idx+1]**2 + Vec[idx]**2)
    return(math.sqrt(tmp))

print('plots!')

###############################
## Plot enrichment functions ##
###############################

plt.figure(1)
for cnt,elem in enumerate(X[0:4]):
    plt.plot(np.linspace(0,1,Xsize), elem/normSqRt(elem,Xsize,dx), label = str(cnt), linewidth = 3) #elem/norm(elem,Xsize,dx)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(2)
for cnt,elem in enumerate(T[0:4]):
    plt.plot(np.linspace(0,0.1,Tsize), elem/normSqRt(elem,Tsize,dt), label = str(cnt), linewidth = 3) #elem/norm(elem,Tsize,dt)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(3)
for cnt,elem in enumerate(K[0:4]):
    plt.plot(np.linspace(1,5,Ksize), elem/normSqRt(elem,Ksize,dk), label = str(cnt), linewidth = 3) #elem/norm(elem,Ksize,dk)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

#################################
## Writeout X,T,K data to file ##
#################################

fileout = open('data.txt','w')
for idx in np.arange(0,Xsize):
    for enr in np.arange(0,len(X)):
        fileout.write(str(X[enr][idx])+'\t')
    fileout.write('\n')

fileout.write('\n\n')
for ids in np.arange(0,Tsize):
    for enr in np.arange(0,len(T)):
        fileout.write(str(T[enr][ids])+'\t')
    fileout.write('\n')

fileout.write('\n\n')
for idk in np.arange(0,Ksize):
    for enr in np.arange(0,len(K)):
        fileout.write(str(K[enr][idk])+'\t')
    fileout.write('\n')

fileout.close()

################################################
## Plot PGD Error as a function of Enrichment ##
################################################

print('\nFinding error of PGD solution as a function of enrichment step.')

def SubLists(a,b):
    c = np.zeros(len(a))
    for idx in np.arange(0,len(a)):
        c[idx] = a[idx]-b[idx]
    return(c)

def normSq(Vec,size,stp):
    tmp = 0.0
    for idx in np.arange(0,size-1):
        tmp += stp/2*(Vec[idx+1]**2 + Vec[idx]**2)
    return(tmp)

def norm(Vec,size,stp):
    tmp = 0.0
    for idx in np.arange(0,size-1):
        tmp += stp/2*(Vec[idx+1] + Vec[idx])
    return(tmp)

ErrTmp1 = np.ones(Tsize)
ErrTmp2 = np.ones((NumEnr,Ksize))
PGDError = np.ones(NumEnr)
lst = os.listdir('./exact/Solutions-200/')
tmpStr = './exact/Solutions-200/'
for idk,filename in enumerate(tqdm(lst)):
    ## obtain exact solution for all space and time for corresponding conductivity value
    ExSoln = np.ones((Tsize,Xsize))
    try:
        with open(tmpStr+filename,'r') as filein:
            for idt,line in enumerate(filein):
                tmp = line.split()
                for idx,elem in enumerate(tmp):
                    ExSoln[idt][idx] = float(elem)
    except FileNotFoundError:
        print(Red+'Text file '+str(tmpStr+item)+' with exact data not found. Please run \"exact.py\" and obtain data. (eventually this will be automated...)')
        sys.exit()
    #######
    for enr in tqdm(np.arange(0,NumEnr)):
        Approx = np.zeros((Tsize,Xsize))
        for idt,t in enumerate(T[enr]):
            for idx,x in enumerate(X[enr]):
                Approx[idt][idx] += t*x*K[enr][idk]
            dum1 = SubLists(ExSoln[idt],Approx[idt]) # gets difference between exact and approximated solution for space at a given time step`
            ErrTmp1[idt] = normSq(dum1,Xsize,dx) # integrate over space to obtain a 1D vector of error as a function of time for that enrichment step and conductivity.
        ErrTmp2[enr,idk] = norm(ErrTmp1,Tsize,dt) # integrate over time to obtain a 2D vector that has error as a function of enrichment and conductivity value

for enr,elem in enumerate(ErrTmp2): # pass each column
    PGDError[enr]=norm(elem,Ksize,dk) # once ErrTmp2 is full, you have the space and time integrated error as a function of conductivity and enrichment step. So integrate out the conductivity and you will have error as a fucntion of enrichment.


plt.figure(4)
plt.semilogy(np.arange(1,NumEnr+1),PGDError)
plt.grid(True)
plt.show()
sys.exit()
