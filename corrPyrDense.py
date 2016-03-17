#import math
import corrConv
import numpy as np
import cv2
import pyrMaskiir
from scipy import ndimage
import os
import time


corr_support=15
#morph_support=int(2.5*corr_support)+1
#morph_support=25
erode_support=2*corr_support
morph_support=erode_support
#erode_support-corr_support
skip_frames=100
confidence=.9
r=1
#confidence = 1-.1/(corr_support*corr_support)
inference_lev=1
nlevs=0
thresh=True

results_dir='./correlate/'

in_name='../Calc2FirstOrderDiffEqSepofVars.mp4'
start_name='calc_2'

#in_name='../noOrangeSampledEvery100.avi'
#start_name='no_orange'
#skip_frames=1

#in_name='../subsamp-change-threshold-0.06-MH12non-euclid.avi'
#start_name='subsamp-change-threshold-0.06-MH12non-euclid'
#skip_frames=1

#in_name='../Quantizing-gravity.mp4'
#start_name='quantizing-grav'

#in_name='../Black-holes-by-Leonard-Susskind.mp4'
#start_name='black-holes'

#in_name='../DiffGeom18FrenetSerretEq.mp4'
#start_name='frenet-seret'


#in_name='../calc2-avg-sample.mp4'
#start_name='calc-2-avg'
#skip_frames=1

#in_name='../preProcessGPU.avi'
#start_name='pre-process'

#in_name='../non-euclid-subsamp-100.avi'
out_name=results_dir+start_name+'-fixed-correlate-'+str(corr_support)+'-mask-lev-'+str(inference_lev)+'-skip-'+str(skip_frames)+'-nlevs-'+str(nlevs)+'-thresh-'+str(thresh)+'-confidence-'+str(confidence)+'-erode-support-'+str(erode_support)+'-iir-'+str(r)

#slides_dir=results_dir+start_name+"-slides"
#os.system("mkdir "+slides_dir)

cap = cv2.VideoCapture(in_name)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#fourcc2 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(out_name+'.mp4',fourcc, fps, (frame_width,frame_height),True)


print (frame_width,frame_height)
tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#tot_time=5000
ret = True

#num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames=tot_time
print num_frames

f=open(out_name+".csv",mode='w')
#f.write("0")
num_channels=3

curr_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
old_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
for lev in range(inference_lev):
    old_frame=cv2.pyrDown(old_frame)
now_frame=np.zeros(old_frame.shape,dtype=np.uint8)
next_frame=np.zeros(old_frame.shape,dtype=np.uint8)
#old_mask=np.zeros(old_frame.shape,dtype=np.uint8)
#new_mask=np.zeros(old_frame.shape,dtype=np.uint8)
mask_frame=np.zeros(old_frame.shape,dtype=np.uint8)
corr_frame=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.float)
#corr_back=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.float)
bit_front=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.bool)
bit_back=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.bool)
bitmask=np.zeros((old_frame.shape[0],old_frame.shape[1]),dtype=np.uint8)

out_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
t=7

#out2=cv2.VideoWriter(out_name+'-mask-'+'.mp4',fourcc2, fps, (old_frame.shape[1],old_frame.shape[0]),True)

def get_new_frame(frame,nlevs):
    band=frame
    for lev in range(nlevs):
        band=cv2.pyrDown(band)
    return band

#first frame
cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);
ret,frame=cap.read()
print ret
pyr_blender=pyrMaskiir.pyrMaskiir(frame,True,nlevs,inference_lev,r)
old_frame[:,:,:]=get_new_frame(frame,inference_lev)
t = t+1

#second frame
cap.set(cv2.CAP_PROP_POS_FRAMES,t*skip_frames);
ret,frame=cap.read()
#pyr_blender=pyrMaskiir.pyrMaskiir(frame,True,nlevs,inference_lev,r)
curr_frame[:,:,:]=frame
now_frame[:,:,:]=get_new_frame(frame,inference_lev)
corr_frame[:,:]=corrConv.fastCorr(old_frame,now_frame,corr_support)
bit_back[:,:]=corr_frame>confidence
tle=0
print tot_time
t = t+1



while(cap.isOpened() and ret):
    print t
    cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*t);
    ret, frame = cap.read()
    next_frame[:,:,:]=get_new_frame(frame,inference_lev)
    tic=time.time()
    corr_frame[:,:]=corrConv.fastCorr(now_frame,next_frame,corr_support)
    print time.time()-tic
    bit_front[:,:]=corr_frame>confidence
    bitmask[:,:]=np.uint8((bit_back & bit_front))*255
    bit_back[:,:]=bit_front
    bitmask[:,:]=cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support)))
    bitmask[:,:]=cv2.erode(bitmask,np.ones((erode_support,erode_support)))
    #bitmask[:,:]=cv2.morphologyEx(cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support))),cv2.MORPH_OPEN,np.ones((morph_support,morph_support)))
    #bitmask[:,:]=
    for color in range(num_channels):
        mask_frame[:,:,color]=bitmask
    cv2.imshow(str(t),(mask_frame*255)*now_frame)
    out_frame[:,:,:]=pyr_blender.maskOut(curr_frame,mask_frame)
    now_frame[:,:,:]=next_frame
    curr_frame[:,:,:]=frame
    
    tle=pyr_blender.get_top_level_energy()
    #if not(started):
    #    bit_frame[:,:]=bit_frame|(bitmask==255)
    #    if np.product(bit_frame):
    #        started=True
    #        print "started"
    #        f.write(str(skip_frames*t)+','+str(tle)+'\n')
    #        out.write(out_frame)
            #out2.write(np.uint8((mask_frame*255)*new_frame))
    #else:
    if(t==200):
        cv2.imwrite('./correlate/bg200.png',out_frame)
        cv2.imwrite('./correlate/mask200.png',mask_frame)
    f.write(str(skip_frames*t)+','+str(tle)+'\n')
    out.write(out_frame)
    #out2.write((mask_frame*255)*new_frame)
    t=t+1
    #old_frame[:,:,:]=new_frame
    if t%10==0:
        cv2.destroyAllWindows()
    if t*skip_frames >= tot_time:
        break




f.close()
cap.release()
cv2.destroyAllWindows()
