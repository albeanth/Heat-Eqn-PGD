import numpy as np
import math
import sys

def updateR(R,X,dx):
    r1=0; r2=0; r3=0; r4=np.zeros(len(X)); r5=np.zeros(len(X))
    for idr in np.arange(0,len(R)-1):
        r1 += dx/2*(math.pow(R[idr+1],2) + math.pow(R[idr],2)) # O(h^2)
        r3 += dx/2*(R[idr+1] + R[idr]) # O(h^2)
        if idr == 0: # first node on LHS
            r2 += (1/(2*dx))*( R[idr+1]*(R[idr+2]-2*R[idr+1]+R[idr]) + 0) # O(h^2)
        elif idr == len(R)-2: # last cell on RHS (calculating between node 59 and 60)
            r2 += (1/(2*dx))*( 0 + R[idr]*(R[idr+1]-2*R[idr]+R[idr-1]) ) # O(h^2)
        else:
            r2 += (1/(2*dx))*( R[idr+1]*(R[idr+2]-2*R[idr+1]+R[idr]) + R[idr]*(R[idr+1]-2*R[idr]+R[idr-1]) ) # O(h^2)
    if len(X)>1:
        for ide,Elem in enumerate(X):
            for cnt in np.arange(0,len(R)-1):
                r5[ide] += dx/2*(R[cnt+1]*Elem[cnt+1] + R[cnt]*Elem[cnt]) # O(h^2)
                if cnt == 0: # first node on LHS
                    r4[ide] += (1/(2*dx))*( R[cnt+1]*(Elem[cnt+2]-2*Elem[cnt+1]+Elem[cnt]) + 0 ) # O(h^2)
                elif cnt == len(R)-2: # next to last node on right side
                    r4[ide] += (1/(2*dx))*( 0 + R[cnt]*(Elem[cnt+1]-2*Elem[cnt]+Elem[cnt-1]) ) # O(h^2)
                else:
                    r4[ide] += (1/(2*dx))*( R[cnt+1]*(Elem[cnt+2]-2*Elem[cnt+1]+Elem[cnt]) + R[cnt]*(Elem[cnt+1]-2*Elem[cnt]+Elem[cnt-1]) ) # O(h^2)
    return(r1,r2,r3,r4,r5)

def updateS(S,T,dt):
    s1=0.0; s2=0.0; s3=0.0; s4=np.zeros(len(T)); s5=np.zeros(len(T)) # note, that the first elem of s4 and s5 will ALWAYS be zero. attributed to initial guess being zeros.
    for ids in np.arange(0,len(S)-1):
        s1 += dt/2*( math.pow(S[ids+1],2) + math.pow(S[ids],2) )# O(h^2)
        s3 += dt/2*( S[ids+1] + S[ids] )# O(h^2)
        if ids == 0: # first node on left side
            s2 += 1/2*( S[ids+1]*(S[ids+1]-S[ids]) + 0 ) # O(h^2)
        else:
            s2 += 1/2*( S[ids+1]*(S[ids+1]-S[ids]) + S[ids]*(S[ids]-S[ids-1]) ) # Backward Euler on both derivatives, O(h)
    if len(T)>1:
        for ide,Elem in enumerate(T):
            for cnt in np.arange(0,len(S)-1):
                s5[ide] += dt/2*( S[cnt+1]*Elem[cnt+1] + S[cnt]*Elem[cnt] ) # O(h^2)
                if cnt == 0: # first node on left side
                    s4[ide] += 1/2*( S[cnt+1]*(Elem[cnt+1]-Elem[cnt]) + 0 ) # O(h^2)
                else:
                    s4[ide] += 1/2*( S[cnt+1]*(Elem[cnt+1]-Elem[cnt]) + S[cnt]*(Elem[cnt]-Elem[cnt-1]) ) # Backward Euler on both derivatives, O(h)
    return(s1,s2,s3,s4,s5)

def updateW(W,K,dk,k):
    w1=0.0; w2=0.0; w3=0.0; w4=np.zeros(len(K)); w5=np.zeros(len(K))
    for idw in np.arange(0,len(W)-1):
        w1 += dk/2*( math.pow(W[idw+1],2) + math.pow(W[idw],2) ) # O(h^2)
        w2 += dk/2*( k[idw+1]*math.pow(W[idw+1],2) + k[idw]*math.pow(W[idw],2) ) # O(h^2)
        w3 += dk/2*( W[idw+1] + W[idw] ) # O(h^2)
    if len(K)>1:
        for ide,Elem in enumerate(K):
            for cnt in np.arange(0,len(W)-1):
                w4[ide] += dk/2*( W[cnt+1]*Elem[cnt+1] + W[cnt]*Elem[cnt] ) # O(h^2)
                w5[ide] += dk/2*( k[cnt+1]*W[cnt+1]*Elem[cnt+1] + k[cnt]*W[cnt]*Elem[cnt] ) # O(h^2)
    return(w1,w2,w3,w4,w5)
