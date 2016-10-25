import sys,os
import numpy as np
import math as m

Xsize = 61
Tsize = m.floor(151/2)
Ksize = m.floor(101/2)

X = np.linspace(0,1,Xsize) # spatial domain -> calculated values for R for first iteration are independent of this first initial guess. tested with parabolic and linear inital shapes and same X_i+1 result was obtained.
T = np.linspace(0,0.1,Tsize) # time domain
K = np.linspace(1,5,Ksize)

fileoutK1 = open('ExSoln-K1.txt','w')
fileoutK5 = open('ExSoln-K5.txt','w')

u = np.zeros((Xsize,Tsize,Ksize))

N = 500
for idx,x in enumerate(X):
    print(x)
    for idt,t in enumerate(T):
        for idk,k in enumerate(K):
            summ = 0.0
            for n in range(1,N+1,2):
                summ += 4./(k * m.pi**3 * n**3) * m.sin(n*m.pi*x) * m.exp(-k * n**2 * m.pi**2 * t)
            u[idx,idt,idk] = x*(1.-x)/(2.*k) - summ

print('\nWriting out data....')
for x in np.arange(0,Xsize):  # prints solution as a function of space for last time point and k=1
    for t in np.arange(0,Tsize):
        # fileout.write('%.6e' % u[x,t,0]+'   ')
        fileoutK1.write(str(u[x,t,0])+'  ')
        fileoutK5.write(str(u[x,t,-1])+'  ')
    fileoutK1.write('\n')
    fileoutK5.write('\n')
fileoutK1.close()
fileoutK5.close()
