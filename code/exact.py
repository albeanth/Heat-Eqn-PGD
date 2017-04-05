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


def exact(space,time,cond, Xsize,Tsize,Ksize):

    u = np.zeros((Ksize,Tsize,Xsize))
    N = 200 # 200 is what chinesta has
    for idk,k in enumerate(tqdm(cond)):
        for idt,t in enumerate(tqdm(time)):
            for idx,x in enumerate(space):
                summ = 0.0
                for n in range(1,N+1,2):
                    summ += 4./(k * m.pi**3 * n**3) * m.sin(n*m.pi*x) * m.exp(-k * n**2 * m.pi**2 * t)
                u[idk,idt,idx] = x*(1.-x)/(2.*k) - summ

    return(u)
