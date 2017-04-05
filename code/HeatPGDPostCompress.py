import numpy as np
import math as m
import sys

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    Yellow = Fore.YELLOW; Red = Fore.RED; Green = Fore.GREEN; Cyan = Fore.CYAN; Magenta = Fore.MAGENTA
    StyDim = Style.DIM
except ImportError:
    print('\nYou should get colorama. It\'s pretty sweet.\n')
    Yellow = ''; Red = ''; Green = ''; Cyan = ''; Magenta = ''
    StyDim = ''

def norm(V,p):
    '''
    Returns the p-norm of a vector.
    '''
    tmp = 0.0
    for elem in V:
        tmp += abs(elem)**p
    tmp = tmp**(1/p)
    return(tmp)

def trapz(V,dv):
    '''
    Trapezoidal rule of integration.
    '''
    tmp = 0.0
    for idv in np.arange(0,len(V)-1):
        tmp += dv/2*(V[idv+1] + V[idv])
    return(tmp)

def updateC(Vc_tmp, Vc, V, dV, NumEnr):
    '''
    Update integrals for compressed PGD algorithm.
    '''
    v1 = 0.0;
    v1 = trapz(np.square(Vc_tmp),dV)

    v3 = np.zeros(len(V))
    for ide,Elem in enumerate(V):
        v3[ide] = trapz(np.multiply(Vc_tmp,Elem),dV)

    if NumEnr == 1:
        v2 = [0.0]
    else:
        v2 = np.zeros(len(Vc))
        for ide,Elem in enumerate(Vc):
            v2[ide] = trapz(np.multiply(Vc_tmp,Elem),dV)

    return(v1, v2, v3)

def PGD_PostComp(eps,nu,Max_fp_iter,MaxEnr,X,T,P,dx,dt,dD,Xsize,Tsize,Dsize):

    Xc = np.array([np.ones(Xsize)],ndmin=2) # spatial domain
    Xc[0][0] = 0.0; Xc[0][-1] = 0.0 # enforcing BCs

    Tc = np.array([np.ones(Tsize)],ndmin=2) # time domain
    Tc[0][0] = 0.0 # enforcing ICs

    Pc = np.array([np.ones(Dsize)],ndmin=2) # diffusivity domain

    NumEnr=0 # number of enrichment steps
    eps_check = 1.0
    fail = 0 # number of enrichment steps that fail to converge
    tmpStr = ''
    while eps_check > 0.0:
        NumEnr += 1 # counter for number of steps for enrichment step convergence

        Xc_old = X[0] # unconverged new enrichment step for space
        Tc_old = T[0]#np.ones(Tsize) # unconverged new enrichment step for time
        Pc_old = P[0]#np.ones(Dsize) # unconverged new enrichment step for parameter (diffusion coeff) space

        enrCnt=0 # number of steps for an enrichment step to converge
        nu_check=1
        while nu_check >= nu:
            enrCnt += 1
            ########################
            ##   Solve for Xc(x)  ##
            ########################
            tc1,tc2,tc3=updateC(Tc_old,Tc,T,dt,NumEnr)
            pc1,pc2,pc3=updateC(Pc_old,Pc,P,dD,NumEnr)
            # print('\ntc -> '+str(tc1)+', '+str(tc2)+', '+str(tc3))
            # print('pc -> '+str(pc1)+', '+str(pc2)+', '+str(pc3))

            Xc_new = np.zeros(Xsize)
            for idx in np.arange(0,Xsize):
                if ((idx==0) or (idx==Xsize-1)): #boundary points
                    Xc_new[idx] = 0.0
                else:
                    tmp = 0.0; tmpc = 0.0
                    for ide,Elem in enumerate(Xc):
                        tmpc += Elem[idx]*tc2[ide]*pc2[ide]
                    for ide,Elem in enumerate(X):
                        tmp += Elem[idx]*tc3[ide]*pc3[ide]
                    Xc_new[idx] = (-tmpc + tmp)/(tc1*pc1)

            # print(Xc_new)

            ########################
            ##   Solve for Tc(t)  ##
            ########################
            xc1,xc2,xc3=updateC(Xc_new,Xc,X,dx,NumEnr)
            # print('xc -> '+str(xc1)+', '+str(xc2)+', '+str(xc3))

            Tc_new = np.zeros(Tsize)
            for idt in np.arange(0,Tsize-1): # from 0 to (Tsize-2)
                tmp = 0.0; tmpc = 0.0
                for ide,Elem in enumerate(Tc):
                    tmpc += Elem[idt+1]*xc2[ide]*pc2[ide]
                for ide,Elem in enumerate(T):
                    tmp += Elem[idt+1]*xc3[ide]*pc3[ide]
                Tc_new[idt+1] = (-tmpc + tmp)/(xc1*pc1)

            # print(Tc_new)
            ########################
            ##   Solve for Pc(D)  ##
            ########################
            tc1,tc2,tc3=updateC(Tc_new,Tc,T,dt,NumEnr)
            # print('tc -> '+str(tc1)+', '+str(tc2)+', '+str(tc3))

            Pc_new = np.zeros(Dsize)
            for idp in np.arange(0,Dsize): # from 0 to (Dsize-1)
                tmp = 0.0; tmpc = 0.0;
                for ide,Elem in enumerate(Pc):
                    tmpc += Elem[idp]*xc2[ide]*tc2[ide]
                for ide,Elem in enumerate(P):
                    tmp += Elem[idp]*xc3[ide]*tc3[ide]
                Pc_new[idp] = (-tmpc + tmp)/(xc1*tc1)

            # print(Pc_new)
            #######################################
            ## Check for enrichment convergence  ##
            #######################################
            intXc_new = trapz(Xc_new,dx)
            intXc_old = trapz(Xc_old,dx)

            intTc_new = trapz(Tc_new,dt)
            intTc_old = trapz(Tc_old,dt)

            intPc_new = trapz(Pc_new,dD)
            intPc_old = trapz(Pc_old,dD)

            new = intXc_new*intTc_new*intPc_new
            old = intXc_old*intTc_old*intPc_old
            nu_check = m.sqrt( (new-old)**2 ) / m.sqrt( old**2 ) #my way - Eq 2.8

            gen = (norm(Xc_new,1)*norm(Tc_new,1)*norm(Pc_new,1))**(1/3)
            Xc_old = Xc_new*gen/norm(Xc_new,1)
            Tc_old = Tc_new*gen/norm(Tc_new,1)
            Pc_old = Pc_new*gen/norm(Pc_new,1)


            # print(StyDim+'      '+str(nu_check))
            # print('{0: .8e},  {1: .8e},  {2: .8e} -> {3: .8e}  ||  {4: .8e},  {5: .8e},  {6: .8e} -> {7: .8e} ==> {8: g}'.format(intXc_old,intTc_old,intPc_old,old, intXc_new,intTc_new,intPc_new,new, nu_check))
            if nu_check < nu:
                print('      Enrichment Step '+str(NumEnr)+' completed in '+str(enrCnt)+' steps.')
            if enrCnt == Max_fp_iter:
                print(Yellow+'      Enrichment step '+str(NumEnr)+' reached the max num of steps.')
                print(StyDim+'      '+str(nu_check))
                fail += 1
                tmpStr += str(NumEnr)+', '
                break
                # sys.exit()


        ##
        ## Enrichment step has converged, tack it onto to Xc, Tc, and Pc and check the overall convergence
        ##
        x_norm = norm(Xc_new,1)
        t_norm = norm(Tc_new,1)
        p_norm = norm(Pc_new,1)

        gen_norm = (x_norm*t_norm*p_norm)**(1/3)
        if NumEnr == 1:
            Xc[0] = Xc_new * gen_norm/x_norm
            Tc[0] = Tc_new * gen_norm/t_norm
            Pc[0] = Pc_new * gen_norm/p_norm
        else:
            Xc = np.vstack((Xc,Xc_new * gen_norm/x_norm))
            Tc = np.vstack((Tc,Tc_new * gen_norm/t_norm))
            Pc = np.vstack((Pc,Pc_new * gen_norm/p_norm))

        ##
        ## Check for overall enrichment convergence on normalized values
        ##
        new = m.sqrt( (intXc_new*intTc_new*intPc_new)**2  )
        old = 0.0
        for enr in np.arange(0,NumEnr): # sums all previous enrichments without the newest enrichment step
            old += trapz(Xc[enr],dx)*trapz(Tc[enr],dt)*trapz(Pc[enr],dD)
        old = m.sqrt(old**2)

        # print(str(new)+', '+str(old))
        eps_check = new/old
        print(Cyan+str(eps_check)+'\n')

        if NumEnr == MaxEnr:
            print(Red+'      Reached max number of enrichment steps')
            break

    print(Yellow+'      '+str(fail)+' enrichment steps ('+str(tmpStr)+') failed to converge to '+str(nu))

    #################################
    ## Writeout X,T,P data to file ##
    #################################

    fileout = open('../data/Compressed_BasisFunc.txt','w')
    for idx in np.arange(0,Xsize):
        for enr in np.arange(0,len(Xc)):
            fileout.write('{0: .8e}   \t'.format(Xc[enr][idx]))
        fileout.write('\n')

    fileout.write('\n\n')
    for idt in np.arange(0,Tsize):
        for enr in np.arange(0,len(Tc)):
            fileout.write('{0: .8e}   \t'.format(Tc[enr][idt]))
        fileout.write('\n')

    fileout.write('\n\n')
    for idp in np.arange(0,Dsize):
        for enr in np.arange(0,len(Pc)):
            fileout.write('{0: .8e}   \t'.format(Pc[enr][idp]))
        fileout.write('\n')

    fileout.close()


    ##  End of Algorithm  ##

    return(Xc,Tc,Pc)
