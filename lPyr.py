import numpy as np
import cv2

def build_l_pyr(im):
    pyr=[]
    while(not(min(im.shape[0],im.shape[1])<20)):
        temp_im=im
        im=cv2.pyrDown(im)
        temp_band=cv2.pyrUp(im)[:temp_im.shape[0],:temp_im.shape[1],:]
        t_band=cv2.subtract(np.array(temp_im,dtype=np.int16),np.array(temp_band,dtype=np.int16))
        pyr.append(t_band)
    pyr.append(im)
    return pyr

def recon_l_pyr(pyr):
    nlevs=len(pyr)
    lowpass=np.array(pyr[nlevs-1],dtype=np.uint8)
    for i in range(nlevs-2,-1,-1):
        band=pyr[i]
        lowpass=cv2.pyrUp(lowpass)[:band.shape[0],:band.shape[1],:]
        highpass=cv2.add(np.array(lowpass,dtype=np.int16),band)
        highpass=cv2.min(highpass,np.array([255,255,255]))
        highpass=cv2.max(highpass,np.array([0,0,0]))
        lowpass=np.array(highpass,dtype=np.uint8)
    return lowpass

def build_l_pyr_vec(im):
    square_pyr=build_l_pyr(im)
    nlevs=len(square_pyr)
    last_ind=0
    pyr=np.array([],dtype=np.int16)
    pind=np.array([[]],dtype=np.int16)
    for lev in range(nlevs):
        band=np.array(square_pyr[lev],dtype=np.int16)
        numel=np.size(band[:,:,0])
        first_ind=last_ind
        last_ind=last_ind+numel
        band=np.reshape(band,(numel,3))
        pyr=np.concatenate(pyr,band)
        pind=np.concatenate(pind,[[first_ind,last_ind]])
    return (pyr,pind)

def recon_l_pyr_vec(pyr,pind):
    return 0
