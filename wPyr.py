import pywt
import numpy as np
def build_w_triangle(signal,wavelet,nlevs):
    temp=signal
    triangle=np.zeros(0,dtype=np.int64)
    trind=np.zeros(0,dtype=np.int64)
    for level in range(nlevs):
        low,high=pywt.dwt(temp,wavelet)
        triangle=np.append(triangle,high)
        trind=np.append(trind,high.shape)
        temp=low
    triangle=np.append(triangle,temp)
    trind=np.append(trind,temp.shape)
    return triangle,trind

def threshold_w_triangle(triangle,trind,thresh):
    return
        
def get_band(triangle,trind,band_num):
    first=np.sum(trind[:band_num])
    last=np.sum(trind[:band_num+1])
    return triangle[first:last]

def set_band(triangle,trind,band,band_num):
    first=np.sum(trind[:band_num])
    last=np.sum(trind[:band_num+1])
    triangle[first:last]=band
    return triangle

def thresh_median(triangle, trind):
    newtri=np.zeros(triangle.shape,dtype=np.int16)
    for b in range(trind.shape[0]-1):
        band=get_band(triangle,trind,b)
        med=np.median(np.abs(band))
        thresh_band=pywt.threshold(band,med)
        print med
        triangle=set_band(triangle,trind,thresh_band,b)
    return triangle

def collapse(triangle,trind,wavelet):
    low=get_band(triangle,trind,trind.shape[0]-1)
    for band in range(1,trind.shape[0]):
        high=get_band(triangle,trind,trind.shape[0]-band-1)
        low=low[:high.shape[0]]
        low=pywt.idwt(low,high,wavelet)
    return low
    
