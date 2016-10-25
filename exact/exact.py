import sys,os
import numpy as np
import math as m
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

try: # try to make new solutions directory
    os.mkdir('Solutions')
except FileExistsError:
    print(Yellow+'\n     Exact solutions seem to already exist.\n')
    sys.exit()

Xsize = 61
Tsize = 151
Ksize = 101

X = np.linspace(0,1,Xsize) # spatial domain -> calculated values for R for first iteration are independent of this first initial guess. tested with parabolic and linear inital shapes and same X_i+1 result was obtained.
T = np.linspace(0,0.1,Tsize) # time domain
K = np.linspace(1,5,Ksize)

u = np.zeros((Xsize,Tsize,Ksize))

N = 500
for idx,x in enumerate(tqdm(X)):
    for idt,t in enumerate(tqdm(T)):
        for idk,k in enumerate(K):
            summ = 0.0
            for n in range(1,N+1,2):
                summ += 4./(k * m.pi**3 * n**3) * m.sin(n*m.pi*x) * m.exp(-k * n**2 * m.pi**2 * t)
            u[idx,idt,idk] = x*(1.-x)/(2.*k) - summ

print('\nWriting out data....\n')
os.chdir('./Solutions')
for idk,k in enumerate(tqdm(K)):
    tmpStr = 'ExSoln-K'+str(k)+'.txt'
    fileout = open(tmpStr,'w')
    for idt in np.arange(0,Tsize):
        for idx in np.arange(0,Xsize):
            fileout.write(str(u[idx,idt,idk])+'\t')
        fileout.write('\n')
    fileout.close()
