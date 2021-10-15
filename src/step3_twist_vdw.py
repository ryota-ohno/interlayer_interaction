import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import Rod
from make import get_monomer_xyzR
from scipy import signal
from scipy.ndimage.filters import maximum_filter

def get_c_vec_vdw(monomer_name,A1,A2,a_,b_,theta):#,name_csv
    
    i=np.zeros(3); a=np.array([a_,0,0]); b=np.array([0,b_,0]); t1=(a+b)/2;t2=(a-b)/2 
    monomer_array_i = get_monomer_xyzR(monomer_name,0.,0.,0.,A1,A2,theta)
    
    monomer_array_t = get_monomer_xyzR(monomer_name,0.,0.,0.,-A1,A2,-theta)
    
    arr_list=[[i,'p'],[b,'p'],[-b,'p'],[a,'p'],[-a,'p'],[t1,'t'],[-t1,'t'],[t2,'t'],[-t2,'t']]##層内の分子の座標とp or t
    Rb_list=[np.round(Rb,1) for Rb in np.linspace(-np.round(b_/2,1),np.round(b_/2,1),int(np.round(2*np.round(b_/2,1)/0.1))+1)]
    Ra_list=[np.round(Ra,1) for Ra in np.linspace(-np.round(a_/2,1),np.round(a_/2,1),int(np.round(2*np.round(a_/2,1)/0.1))+1)]
    z_list=[];V_list=[];z_2dlist=[]
    for Ra in Ra_list:
        z_maxlist=[]
        for Rb in Rb_list:
            z_max=0
            for R,arr in arr_list:
                if arr=='t':
                    monomer_array1=monomer_array_t
                elif arr=='p':
                    monomer_array1=monomer_array_i
                for x1,y1,z1,R1 in monomer_array1:#層内
                    x1,y1,z1=np.array([x1,y1,z1])+R
                    for x2,y2,z2,R2 in monomer_array_i:#i0
                        x2+=Ra
                        y2+=Rb
                        z_sq=(R1+R2)**2-(x1-x2)**2-(y1-y2)**2
                        if z_sq<0:
                            z_clps=0.0
                        else:
                            z_clps=np.sqrt(z_sq)+z1-z2
                        z_max=max(z_max,z_clps)
            z_list.append(z_max)
            V_list.append(a_*b_*z_max)
            z_maxlist.append(-z_max)
        z_2dlist.append(z_maxlist)
    #return np.array([Ra_list,Rb_list,z_list])
    return z_2dlist
    
def detect_peaks(image, filter_size=3, order=0.5):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image, mask=~(image == local_max))
    return detected_peaks
