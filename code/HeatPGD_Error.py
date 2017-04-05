import sys
import os
import numpy as np
import math as m
import matplotlib.pyplot as plt

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    Yellow = Fore.YELLOW; Red = Fore.RED; Green = Fore.GREEN; Cyan = Fore.CYAN; Magenta = Fore.MAGENTA
    StyDim = Style.DIM
except ImportError:
    print('\nYou should get colorama. It\'s pretty sweet.\n')
    Yellow = ''; Red = ''; Green = ''; Cyan = ''; Magenta = ''
    StyDim = ''

def IntOut(FullErr, dx,dt,dk):
    size = FullErr.shape

    ## integrate out space
    tmp = np.zeros((size[0],size[1]))
    for x in np.arange(0,size[0]):
        for y in np.arange(0,size[1]):
            tmp[x,y] = trapz(FullErr[x,y,:],dx)

    ## integrate out time
    tmp2 = np.zeros(size[0])
    for x in np.arange(0,size[0]):
        tmp2[x] = trapz(tmp[x,:],dt)

    ## integrate out conductivity
    final = trapz(tmp2,dk)
    return(final)

def trapz(V,dv):
    '''
    Trapezoidal rule of integration.
    '''
    tmp = 0.0
    for idv in np.arange(0,len(V)-1):
        tmp += dv/2*(V[idv+1] + V[idv])
    return(tmp)


def XD_Error(space,cond, X,T,K, Xc,Tc,Kc, Xsize,Tsize,Ksize, dx,dt,dk, ExactSoln, figCnt):

    TotNumEnr = len(X)

    PGD_Sol = np.zeros((Ksize,Tsize,Xsize)) # PGD solution as a function of space and enrichment step for all time and D=2.5
    PGD_Sol_Tot = np.zeros((Ksize,Tsize,Xsize)) # PGD solution as a function of space and enrichment step for all time and D=2.5
    FullErr_R = np.zeros((Ksize,Tsize,Xsize)) # PGD solution as a function of space for all enrichment steps, all time, and D=2.5
    TimeErr_R = np.zeros((Ksize,Tsize,TotNumEnr))
    PGD_Error = np.ones((TotNumEnr,2)) # col 1 is 2-norm error, col 2 is max

    PGDc_Sol = np.zeros((Ksize,Tsize,Xsize)) # PGD solution as a function of space and enrichment step for all time and D=2.5
    PGDc_Sol_Tot = np.zeros((Ksize,Tsize,Xsize)) # PGD solution as a function of space and enrichment step for all time and D=2.5
    FullErr_C = np.zeros((Ksize,Tsize,Xsize)) # PGD solution as a function of space for all enrichment steps, all time, and D=2.5
    TimeErr_C = np.zeros((Ksize,Tsize,TotNumEnr))
    PGDc_Error = np.ones((TotNumEnr,2)) # col 0 is integrated error, col 1 is max

    # plt.figure(figCnt); figCnt+=1
    # TimeErrR_Out = open('../data/PGDRaw_TimeError.txt','w')
    # TimeErrC_Out = open('../data/PGDComp_TimeError.txt','w')
    PGDEnrError = open('../data/PGD_Enr_Error.txt','w')
    PGD_Enr_Error = np.ones((TotNumEnr,2)) # col 0 is raw, col 1 is compressed

    for EnrStep in np.arange(0,TotNumEnr):
        for idk in np.arange(0,Ksize):
            for idt in np.arange(0,Tsize):
                for idx in np.arange(0,Xsize):
                    # get PGD solutions for enrichement step
                    PGD_Sol[idk,idt,idx] = X[EnrStep][idx] * T[EnrStep][idt] * K[EnrStep][idk]
                    PGDc_Sol[idk,idt,idx] = Xc[EnrStep][idx] * Tc[EnrStep][idt] * Kc[EnrStep][idk]

        # get total PGD solution, sum over enr steps
        PGD_Sol_Tot += PGD_Sol
        PGDc_Sol_Tot += PGDc_Sol
        # get errors for enrichment step at all time steps for all space for all conductivity
        FullErr_R  = np.square(np.subtract( ExactSoln, PGD_Sol_Tot))
        FullErr_C = np.square(np.subtract( ExactSoln, PGDc_Sol_Tot))

        PGD_Enr_Error[EnrStep,0] = IntOut(FullErr_R,dx,dt,dk)
        PGD_Enr_Error[EnrStep,1] = IntOut(FullErr_C,dx,dt,dk)
        PGDEnrError.write('{0: g}\t{1: .6e}\t{2: .6e}\n'.format(EnrStep+1, PGD_Enr_Error[EnrStep,0], PGD_Enr_Error[EnrStep,1]))
        #
        # # Get max error of FullErr_R/C
        # intTimeR = np.ones(Tsize); intTimeC = np.ones(Tsize)
        # for idt,Telem in enumerate(FullErr_R):
        #     # print('idt = {0: g}, -> {1: .5e}'.format(idt,np.max(Telem)))
        #     TimeErr_R[idt,EnrStep] = np.max(Telem) # get max error value in time
        #     intTimeR[idt] = trapz(Telem,dx) # integrate each time each step across space
        #
        # for idt,Telem in enumerate(FullErr_C):
        #     # print('idt = {0: g}, -> {1: .5e}'.format(idt,np.max(Telem)))
        #     TimeErr_C[idt,EnrStep] = np.max(Telem) # get max error value in time
        #     intTimeC[idt] = trapz(Telem,dx) # integrate each time each step across space
        #
        #
        # PGD_Error[EnrStep,0] = trapz(intTimeR,dt) # integrate over time
        # PGD_Error[EnrStep,1] = np.max(TimeErr_R[:,EnrStep])
        # PGDc_Error[EnrStep,0] = trapz(intTimeC,dt)
        # PGDc_Error[EnrStep,1] = np.max(TimeErr_C[:,EnrStep])
        # plt.plot(space, PGDc_Sol[Tindex,:], linewidth = 3, label = 'EnrStep = '+str(EnrStep+1))


    # for idt in np.arange(0,Tsize):
    #     TimeErrR_Out.write('{0:g} \t '.format(idt/Tsize))
    #     TimeErrC_Out.write('{0:g} \t '.format(idt/Tsize))
    #     for EnrStep in np.arange(0,TotNumEnr):
    #         TimeErrR_Out.write('{0: .6e} \t '.format(TimeErr_R[idt,EnrStep]))
    #         TimeErrC_Out.write('{0: .6e} \t '.format(TimeErr_C[idt,EnrStep]))
    #     TimeErrR_Out.write('\n')
    #     TimeErrC_Out.write('\n')
    #
    # plt.plot(space, PGDc_Sol_Tot[Tindex,:], linewidth = 3, label = 'Total PGDc')
    # plt.plot(space, ExactSoln[Dval][Tindex,:], linewidth = 3, label = 'Ref Soln')

    # plt.grid(True)
    # plt.legend(loc='best',fontsize='x-small')
    # plt.xlabel('space domain')

    # print('                Integrated Error                   Max Error')
    # print('EnrStep         Raw         Comp.             Raw             Comp.')
    # dum = 1
    # for pair in zip(PGD_Error, PGDc_Error):
    #     print('{0: g}\t{1: .6e}, {2: .6e} \t {3: .6e}, {4: .6e} '.format(dum, pair[0][0],pair[1][0], pair[0][1],pair[1][1]))
    #     dum+=1

    plt.figure(figCnt); figCnt+=1
    plt.semilogy(np.arange(1,TotNumEnr+1), PGD_Enr_Error[:,0], linewidth = 3, label = 'Raw PGD')
    plt.semilogy(np.arange(1,TotNumEnr+1), PGD_Enr_Error[:,1], linewidth = 3, label = 'Comp PGD')
    plt.title('Integrated PGD Error')
    plt.xlabel('Enrichment Number')
    plt.ylabel('2-norm difference')
    plt.legend()
    plt.grid(True)

    # plt.figure(figCnt); figCnt+=1
    # plt.semilogy(np.divide(np.arange(1,Tsize),Tsize), TimeErr_R[1:,-1], linewidth = 3, label = 'Raw PGD')
    # plt.semilogy(np.divide(np.arange(1,Tsize),Tsize), TimeErr_C[1:,-1], linewidth = 3, label = 'Comp PGD')
    # plt.xlabel('Time')
    # plt.ylabel('max((U_ex - U_PGD)^2)')
    # plt.legend(loc = 0)
    # plt.grid(True)

    plt.show()
