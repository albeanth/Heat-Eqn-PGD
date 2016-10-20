import numpy as np
import math

def updateR(R,X,dx):
    r1=0; r2=0; r3=0; r4=0; r5=0
    for idr,dum in enumerate(R):
        if idr == len(R)-1: #far right boundary node
            pass
        else:
            r1 += dx/2*(math.pow(R[idr+1],2) + math.pow(R[idr],2)) # O(h^2)
            r3 += dx/2*(R[idr+1] + R[idr]) # O(h^2)
            if idr == 0: # first node on LHS
                r2 += 1/(2*dx)*( R[idr+1]*(R[idr+2]-2*R[idr+1]+R[idr]) + 0) # O(h^2)
                # r2 += 1/(2*dx)*( R[idr+1]*(R[idr+2]-2*R[idr+1]+R[idr]) + R[idr]/4*(R[idr]-2*R[idr+1]+R[idr+2]) ) # O(h)
            elif idr == len(R)-2: # next to last node on right side
                r2 += 1/(2*dx)*( 0 + R[idr]*(R[idr+1]-2*R[idr]+R[idr-1]) ) # O(h^2)
                # r2 += 1/(2*dx)*( R[idr+1]*(R[idr-1]-2*R[idr]+R[idr+1]) + R[idr]*(R[idr+1]-2*R[idr]+R[idr-1]) ) # O(h)
            else:
                r2 += 1/(2*dx)*( R[idr+1]*(R[idr+2]-2*R[idr+1]+R[idr]) + R[idr]*(R[idr+1]-2*R[idr]+R[idr-1]) ) # O(h^2)
    for Elem in X:
        for cnt,pair in enumerate(zip(Elem,R)):
            if cnt == len(R)-1: # far right boundary cell
                pass
            else:
                r5 += dx/2*(R[cnt+1]*Elem[cnt+1] + R[cnt]*Elem[cnt]) # O(h^2)
                if cnt == 0: # first node on LHS
                    r4 += 1/(2*dx)*( R[cnt+1]*(Elem[cnt+2]-2*Elem[cnt+1]+Elem[cnt]) + 0 ) # O(h^2)
                    # r4 += 1/(2*dx)*( R[cnt+1]*(Elem[cnt+2]-2*Elem[cnt+1]+Elem[cnt]) + R[cnt]/4*(Elem[cnt]-2*Elem[cnt+1]+Elem[cnt+2]) ) # O(h)
                elif cnt == len(R)-2: # next to last node on right side
                    r4 += 1/(2*dx)*( 0 + R[cnt]*(Elem[cnt+1]-2*Elem[cnt]+Elem[cnt-1]) ) # O(h^2)
                    # r4 += 1/(2*dx)*( R[cnt+1]*(Elem[cnt-1]-2*Elem[cnt]+Elem[cnt+1]) + R[cnt]*(Elem[cnt+1]-2*Elem[cnt]+Elem[cnt-1]) ) # O(h)
                else:
                    r4 += 1/(2*dx)*( R[cnt+1]*(Elem[cnt+2]-2*Elem[cnt+1]+Elem[cnt]) + R[cnt]*(Elem[cnt+1]-2*Elem[cnt]+Elem[cnt-1]) ) # O(h^2)
    return(r1,r2,r3,r4,r5)

def updateS(S,T,dt):
    s1=0.0; s2=0.0; s3=0.0; s4=0.0; s5=0.0
    for ids,dum in enumerate(S):
        if ids == len(S)-1:
            pass
        else:
            s1 += dt/2*( math.pow(S[ids+1],2) + math.pow(S[ids],2) )# O(h^2)
            s3 += dt/2*( S[ids+1] + S[ids] )# O(h^2)
            if ids == 0: # first node on left side
                s2 += 1/2*( S[ids+1]*(S[ids+1]-S[ids]) + 0 ) # O(h^2)
                # s2 += 1/2*( S[ids+1]*(S[ids+1]-S[ids]) + S[ids]*(S[ids+1]-S[ids]) ) # Forward Euler on second derivative,  O(h)
            else:
                s2 += 1/2*( S[ids+1]*(S[ids+1]-S[ids]) + S[ids]*(S[ids]-S[ids-1]) ) # Backward Euler on both derivatives, O(h)
    for Elem in T:
        for cnt,pair in enumerate(zip(Elem,S)):
            if cnt == len(S)-1:
                pass
            else:
                s5 += dt/2*( S[cnt+1]*Elem[cnt+1] + S[cnt]*Elem[cnt] ) # O(h^2)
                if cnt == 0: # first node on left side
                    s4 += 1/2*( S[cnt+1]*(Elem[cnt+1] - Elem[cnt]) + S[cnt]*(Elem[cnt+1] - Elem[cnt]) ) # Forward Euler on second derivative,  O(h)
                else:
                    s4 += 1/2*( S[cnt+1]*(Elem[cnt+1] - Elem[cnt]) + S[cnt]*(Elem[cnt] - Elem[cnt-1]) ) # Backward Euler on both derivatives, O(h)
    return(s1,s2,s3,s4,s5)

def updateW(W,K,dk,k):
    w1=0.0; w2=0.0; w3=0.0; w4=0.0; w5=0.0
    for idw,dum in enumerate(W):
        if idw == len(W)-1:
            pass
        else:
            w1 += dk/2*( math.pow(W[idw+1],2) + math.pow(W[idw],2) ) # O(h^2)
            w2 += k[idw]*dk/2*( math.pow(W[idw+1],2) + math.pow(W[idw],2) ) # O(h^2)
            w3 += dk/2*( W[idw+1] + W[idw] ) # O(h^2)
    for Elem in K:
        for cnt,pair in enumerate(zip(Elem,W)):
            if cnt == len(W)-1:
                pass
            else:
                w4 += dk/2*( W[cnt+1]*Elem[cnt+1] + W[cnt]*Elem[cnt] ) # O(h^2)
                w5 += k[cnt]*dk/2*( W[cnt+1]*Elem[cnt+1] + W[cnt]*Elem[cnt] ) # O(h^2)
    return(w1,w2,w3,w4,w5)


# def OLD_updateR(R,X,dx):
#     r1=0; r2=0; r3=0; r4=0; r5=0
#     for idr,dum in enumerate(R):
#         if ((idr==0) or (idr==len(R)-1)): # boundary points
#             pass
#         else: # interior points
#             r1 += dx/2 * (math.pow(R[idr],2)+math.pow(R[idr-1],2))
#             if idr == 1:
#                 r2 += math.pow(dx,3)/8 * (R[idr]*(R[idr]-2*R[idr+1]+R[idr]) + R[idr-1]*(R[idr-1]-2*R[idr]+R[idr+1]))
#             else:
#                 r2 += 1/4 * (R[idr]*(R[idr+1]-2*R[idr]+R[idr-1]) + R[idr-1]*(R[idr]-2*R[idr-1]+R[idr-2]))
#             r3 += dx/2 * (R[idr]+R[idr-1])
#     for Elem in X:
#         for cnt,pair in enumerate(zip(Elem,R)):
#             if ((cnt==0) or (cnt==np.shape(X)[1]-1)): #boundary points
#                 pass
#             else: #interior points
#                 if cnt ==1:
#                     r4 += math.pow(dx,3)/8 * (R[cnt]*(Elem[cnt]-2*Elem[cnt+1]+Elem[cnt]) + R[cnt-1]*(Elem[cnt-1]-2*Elem[cnt]+Elem[cnt+1]))
#                 else:
#                     r4 += 1/4 * (R[cnt]*(R[cnt+1]-2*R[cnt]+R[cnt-1]) + R[cnt-1]*(R[cnt]-2*R[cnt-1]+R[cnt-2]))
#                 r5 += dx/2 * (R[cnt]*Elem[cnt]+R[cnt-1]*Elem[cnt-1])
#     return(r1,r2,r3,r4,r5)
# def OLD_updateS(S,T,dt):
#     s1=0.0; s2=0.0; s3=0.0; s4=0.0; s5=0.0
#     for ids,dum in enumerate(S):
#         if ((ids==0) or (ids==len(S)-1)): # boundary points
#             pass
#         else: # interior points
#             s1 += dt/2 * (math.pow(S[ids],2)+math.pow(S[ids-1],2))
#             s2 += 1/4 * (S[ids]*(S[ids+1]-S[ids]) + S[ids-1]*(S[ids]-S[ids-1]))
#             s3 += dt/2 * (S[ids]+S[ids-1])
#     for Elem in T:
#         for cnt,pair in enumerate(zip(Elem,S)):
#             if ((cnt==0) or (cnt==np.shape(T)[1]-1)): #boundary points
#                 pass
#             else: #interior points
#                 s4 += 1/4 * (S[cnt]*(Elem[cnt+1]-Elem[cnt]) + S[cnt-1]*(Elem[cnt]-Elem[cnt-1]))
#                 s5 += dt/2 * (S[cnt]*Elem[cnt] + S[cnt-1]*Elem[cnt-1])
#     return(s1,s2,s3,s4,s5)
# def OLD_updateW(W,K,dk,k):
#     w1=0.0; w2=0.0; w3=0.0; w4=0.0; w5=0.0
#     for idw,dum in enumerate(W):
#         if ((idw==0) or (idw==np.shape(K)[1]-1)):
#             pass
#         else:
#             w1 += dk/2 * (math.pow(W[idw],2)+math.pow(W[idw-1],2))
#             w2 += k * dk/2 * (math.pow(W[idw],2)+math.pow(W[idw-1],2))
#             w3 += dk/2 * (W[idw]+W[idw-1])
#     for Elem in K:
#         for cnt,pair in enumerate(zip(Elem,W)):
#             if ((cnt==0) or (cnt==np.shape(K)[1]-1)): #boundary points
#                 pass
#             else:
#                 w4 += dk/2 * (W[cnt]*Elem[cnt]+W[cnt-1]*Elem[cnt-1])
#                 w5 += k * dk/2 * (W[cnt]*Elem[cnt]+W[cnt-2]*Elem[cnt-2])
#     return(w1,w2,w3,w4,w5)
