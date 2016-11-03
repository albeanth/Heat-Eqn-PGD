import sys,os
import os.path
import numpy as np
import math
import matplotlib.pyplot as plt
import basis
import linecache

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

###############################################################################

X = np.array([np.zeros(Xsize)],ndmin=2) # spatial domain -> calculated values for R for first iteration are independent of this first initial guess. tested with parabolic and linear inital shapes and same X_i+1 result was obtained.
T = np.array([np.zeros(Tsize)],ndmin=2) # time domain
K = np.array([np.zeros(Ksize)],ndmin=2) # diffusivity domain
dx = (1-0)/Xsize
dt = (0.1-0)/Tsize
dk = (5-1)/Ksize

## guesses for R,S,W
R_old = np.ones(Xsize)
R_old[0] = 0; R_old[-1] = 0

S_old = np.ones(Tsize)
S_old[0] = 0
S_new = np.ones(Tsize)
S_new[0] = 0

W_old = np.ones(Ksize)
W_new = np.ones(Ksize)

## convergence criteria
eps = 1E-4 # num of enrichment steps - 2-norm
nu = 1E-5 # within enrichemnt - inf-norm

NumEnr=0 # number of enrichment steps
eps_check = 1.0
while eps_check > eps:
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
        # R_new = linalg.solve_triangular(R_Lmatrix,bVec)

        #######################
        ##   Solve for S(t)  ##
        #######################
        # print('Solving for S(t)')
        r1,r2,r3,r4,r5 = basis.updateR(R_new,X,dx)
        # print('updatedR -> '+str(r1)+', '+str(r2)+', '+str(r3)+', '+str(r4)+', '+str(r5))
        # All 'w' integrals already solved for in R(x) solve. Don't need to resolve them.

        # for idy in np.arange(0,Tsize):
        #     if idy == 0:
        #         S_new[idy] = 0.0
        #     else:
        #         tmp = 0.0
        #         for ide,Elem in enumerate(T):
        #             tmp += r5[ide]*w4[ide]*(Elem[idy]-Elem[idy-1])/dt - Elem[idy]*w5[ide]*r4[ide]
        #         S_new[idy] = (S_new[idy-1]*w1*r1/dt - tmp + w3*r3*f)/(w1*r1/dt - w2*r2)
        # print(S_new)

        for idy in np.arange(0,Tsize-1):
            tmp = 0.0
            for ide,Elem in enumerate(T):
                tmp += r5[ide]*w4[ide]*(Elem[idy+1]-Elem[idy])/dt - Elem[idy]*w5[ide]*r4[ide]
            S_new[idy+1] = (S_new[idy]*w1*r1/dt - tmp + w3*r3*f)/(w1*r1/dt - w2*r2)

        #######################
        ##   Solve for W(k)  ##
        #######################
        # print('Solving for W(k)')
        # All 'r' integrals already solved for in R(x) solve. Don't need to resolve them.
        s1,s2,s3,s4,s5 = basis.updateS(S_new,T,dt)
        # print('updatedS -> '+str(s1)+', '+str(s2)+', '+str(s3)+', '+str(s4)+', '+str(s5))

        for idk,param in enumerate(k): # each element in conductivity range
            tmp = 0.0
            for ide,Elem in enumerate(K): # each enrichment step
                # print(Elem[idk],r5[ide],s4[ide],r4[ide],s5[ide])
                tmp += Elem[idk]*r5[ide]*s4[ide] - param*Elem[idk]*r4[ide]*s5[ide]
            # print(tmp, (f*r3*s3), r1*s2-param*s1*r2)
            W_new[idk] = (-tmp + f*r3*s3)/(r1*s2-param*s1*r2)
        # print(W_new)
        #######################################
        ## Check for enrichment convergence  ##
        #######################################

        nu_check = 0.0
        for idr in np.arange(0,Xsize):
            for ids in np.arange(0,Tsize):
                for idw in np.arange(0,Ksize):
                    tmp = abs((R_new[idr]*S_new[ids]*W_new[idw] - R_old[idr]*S_old[ids]*W_old[idw]))
                    if tmp > nu_check:
                        nu_check = tmp
                    else:
                        pass

        print(Cyan+str(nu_check))
        enrCnt += 1
        # sys.exit()
        if nu_check >= nu:
            R_old = R_new
            S_old = S_new
            W_old = W_new
            # if enrCnt == 1:
            #     print(Red+'maximum iteration count reached.')
            #     X = np.vstack((X,R_new))
            #     T = np.vstack((T,S_new))
            #     K = np.vstack((K,W_new))
            #     break
        else:
            NumEnr += 1 # counter for number of steps for enrichment step convergence
            print(Green+'Enrichment Step '+str(NumEnr)+' completed in '+str(enrCnt)+' steps.')
            X = np.vstack((X,R_new))
            T = np.vstack((T,S_new))
            K = np.vstack((K,W_new))

    print(Yellow+'   ....checking enrichment convergence')

    TotSum = 0.0; NewSum = 0.0
    for enr in np.arange(0,len(X)):
        for idr in np.arange(0,Xsize):
            for idt in np.arange(0,Tsize):
                for idk in np.arange(0,Ksize):
                    TotSum += X[enr][idr]*T[enr][idt]*K[enr][idk]
                    if enr == len(X)-1:
                        NewSum += X[enr][idr]*T[enr][idt]*K[enr][idk]

    eps_check = abs(NewSum/TotSum)
    print(Magenta+'   '+str(eps_check))
    # sys.exit()
    if eps_check >= eps:
        if NumEnr == 3:
            break
        R_old = X[-1]
        S_old = T[-1]
        W_old = K[-1]
    else:
        print(Green+'Done. It took '+str(NumEnr)+ ' enrichement steps to converge solution.\n')
        break

###############################################################################

def maxim(a):
    maxx = 0
    for elem in a:
        if abs(elem)>maxx:
            maxx = abs(elem)
    return(maxx)

print('plots!')
plt.figure(1)
for cnt,elem in enumerate(X):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(0,1,Xsize), -elem/np.linalg.norm(elem,2), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(2)
for cnt,elem in enumerate(T):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(0,0.1,Tsize), elem/np.linalg.norm(elem,2), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')

plt.figure(3)
for cnt,elem in enumerate(K):
    if cnt == 0:
        pass
    else:
        plt.plot(np.linspace(1,5,Ksize), -elem/np.linalg.norm(elem,2), label = str(cnt), linewidth = 3)
plt.grid(True)
plt.legend(loc='best',fontsize='x-small')
plt.show()

sys.exit()




print('\nFinding error of PGD solution as a function of enrichement step.')

# PGDError = np.ones(NumEnr)
# tmp = np.ones(NumEnr)
# lst = os.listdir('./exact/Solutions/')
# tmpStr = './exact/Solutions/'
# for idx,item in enumerate(lst):
#     ExSoln = np.ones((Tsize,Xsize))
#     try:
#         with open(tmpStr+item,'r') as file1:
#             for cnt,line in enumerate(file1):
#                 ExSoln[cnt] = line.split()
#                 for idx,elem in enumerate(ExSoln[cnt]):
#                     ExSoln[cnt][idx] = float(elem)
#
#     except FileNotFoundError:
#         print(Red+'Text file '+str(tmpStr+item)+' with exact data not found. Please run \"exact.py\" and obtain data. (eventually this will be automated...)')
#         sys.exit()

PGDError = np.ones(NumEnr)
Approx = np.ones(Tsize)
lst = os.listdir('./exact/Solutions/')
tmpStr = './exact/Solutions/'
for ide in np.arange(0,len(X)):
    for idk,k in enumerate(K[:ide+1]):
        ## obtain exact solution for all space and time for corresponding conductivity value
        ExSoln = np.ones((Tsize,Xsize))
        try:
            with open(tmpStr+lst[idk],'r') as file1:
                for cnt,line in enumerate(file1):
                    ExSoln[cnt] = line.split()
                    for idx,elem in enumerate(ExSoln[cnt]):
                        ExSoln[cnt][idx] = float(elem)
        except FileNotFoundError:
            print(Red+'Text file '+str(tmpStr+item)+' with exact data not found. Please run \"exact.py\" and obtain data. (eventually this will be automated...)')
            sys.exit()
        ##
        for idt,t in enumerate(T[:ide+1]):
            for idx,x in enumerate(X[:ide+1]):
                pass
        PGDError[ide] += np.linalg.norm((ExSoln-Approx),2)

plt.figure(1)
plt.plot(np.arange(1,len(PGDError_K1)+1),PGDError)
plt.grid(True)
plt.show()
