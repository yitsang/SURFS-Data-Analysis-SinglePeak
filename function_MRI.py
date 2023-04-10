#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:42:59 2023
This Code is based on Te-wei Tsai's code on python 2 environment
Only transfer the code from python 2 to 3 for easier use
No more modifications in this code version
@author: Yi Zeng
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import os
from scipy.optimize import leastsq

def load_file(filepath):
    data_signal = np.fromfile(filepath, sep=" ")
    n = data_signal.size  # Check the length of data
    data_base = data_signal.reshape((n // 2, 2))  # Reshape the data to a matrix

    data_positon_base = data_base[:, 0]
    data_base = data_base[:, 1]

    # Take the data of move back
    back_index = np.diff(data_positon_base) < 0
    back_index = np.append(back_index, back_index[back_index.size - 1])

    x = data_positon_base[back_index]
    y = data_base[back_index]

    filename = os.path.basename(filepath)

    return x, y, filename


def signal_noise_ratio(path_ref, nn):
    # The inputs are the file path and the number of point to take

    data_ref = np.fromfile(path_ref, sep=" ")
    n = data_ref.size
    data_ref = data_ref.reshape((n // 2, 2))
    temp_ref = data_ref[:, 1]
    temp_ref = temp_ref[n // 2 - nn:n // 2]  # The square wave

    center = (max(temp_ref) + min(temp_ref)) / 2
    crest = np.mean(temp_ref[temp_ref > center])
    valley = np.mean(temp_ref[temp_ref < center])
    snr_height = crest - valley

    filename = os.path.basename(path_ref)

    return temp_ref, snr_height, filename


def position_filter(temp_x1, temp_y1, temp_x0, temp_y0, cen_peak, space_peak):
    # The data in the range
    temp_x0_center = temp_x0[(temp_x0 >= cen_peak - space_peak) &
                             (temp_x0 <= cen_peak + space_peak)]
    temp_y0_center = temp_y0[(temp_x0 >= cen_peak - space_peak) &
                             (temp_x0 <= cen_peak + space_peak)]

    # The intersection of signal and baseline
    temp_x1_center = temp_x1[(temp_x1 >= cen_peak - space_peak) &
                             (temp_x1 <= cen_peak + space_peak)]
    temp_y1_center = temp_y1[(temp_x1 >= cen_peak - space_peak) &
                             (temp_x1 <= cen_peak + space_peak)]

    # Take the fitting value for the same x value
    temp_y0_fit = np.interp(temp_x1_center,
                            np.flipud(temp_x0_center), np.flipud(temp_y0_center))

    # Take the difference to get the signal
    signal_x = temp_x1_center
    signal_y = temp_y1_center - temp_y0_fit

    return signal_x, signal_y

def airPLS(x, lambda_base, order, wep, p, itermax):
    
    # This method needs to cite:  
    # Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive 
    # iteratively reweighted penalized least squares. 
    # Analyst 135 (5), 1138-1146 (2010).
        
    n = np.size(x)
    wi = np.append(np.arange(np.ceil(n*wep),dtype=int), 
                   np.arange(np.floor(n-n*wep)-1,n,dtype=int))

    D = sp.csc_matrix(np.eye(n))
    for num in range (0,order):
        D = D[1:]-D[:-1]
    
    DD = lambda_base*D.T*D
    
    # Begin the iteration
    w = np.ones(n)      # Initial guess
    for jj in range (1,itermax+1):
      
        W = sp.csc_matrix((w,(np.arange(w.size),np.arange(w.size))), 
                          shape=(w.size,w.size))
      
        C = W+DD               
        z =spl.spsolve(C,w*x)       
        d = x-z;

        dssn = abs(sum(d[d<0]))
          
        if (dssn < 0.001*sum(abs(x))):  # problem 
            break

        w[d>=0] = 0
        w[wi] = p
        w[d<0] = jj*np.exp(abs(d[d<0])/dssn)
        
    xc = x-z    

    return xc, z

def bfvar_temp(cut_positions, x, y):
    
    a1 = np.int(np.min(cut_positions))
    a2 = np.int(np.max(cut_positions))
    
    # 20 is the space of points to take. This value can be tuned.    
    pts = np.append(np.arange(0,a1,20,dtype=int),
                    np.arange(a2,x.size,20,dtype=int))

    # Make sure to get the start and end points
    if pts[pts.size-1] != x.size-1:
       pts = np.append(pts,x.size-1)
      
    # Flip the data for the interpolation  
    yt = np.interp(np.arange(x.size),pts,y[pts])   
    
    return yt

def error_bar_ana(index, y):
    
    index_max = np.empty(0)  # local max
    index_min = np.empty(0)  # local min
    for ii in range(1,index.size-2):
    
        if (y[index[ii]]>y[index[ii-1]])& \
           (y[index[ii]]>y[index[ii+1]]):
            index_max = np.append(index_max,index[ii])
        elif (y[index[ii]]<y[index[ii-1]])& \
             (y[index[ii]]<y[index[ii+1]]):
             index_min = np.append(index_min,index[ii])
        
    index_max = index_max.astype(int)
    index_min = index_min.astype(int)
    
    return index_max, index_min

def B_field_fit(x,p):
    
    # This function is to fit the magnetic field.    
    angle_m, M, signal_base, d, d_x = p    
  
    # Alignment of magnetic dipole
    x_m = np.sin(angle_m)
    y_m = np.cos(angle_m)
    
    # Magnetic field        
    r = np

def B_field_fit(x,p):
    
    # This function is to fit the magnetic field.    
    angle_m, M, signal_base, d, d_x = p    
  
    # Alignment of magnetic dipole
    x_m = np.sin(angle_m)
    y_m = np.cos(angle_m)
    
    # Magnetic field        
    r = np.sqrt(np.power(d_x-x,2) + np.power(d,2))
    cos_theta = ((d_x-x)*x_m+d*y_m)/r
    Bz = M/np.power(r,3)*(3*np.power(cos_theta,2) - 1) + signal_base

    return Bz
    
def residuals(p, y, x): 
    return y - B_field_fit(x, p) 

