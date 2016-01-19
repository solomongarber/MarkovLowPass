#import math
import corrConv
import numpy as np
import cv2
import pyrMask
from scipy import ndimage
import os
import time


corr_support=12
morph_support=int(2.5*corr_support)+1
#morph_support=25
skip_frames=100
#confidence=.9995
confidence = 1-.1/(corr_support*cor_support)
nlevs=0
thresh=True

results_dir='./correlate/'

in_name='../Calc2FirstOrderDiffEqSepofVars.mp4'
start_name='calc_2'

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
out_name=results_dir+start_name+'-correlate-'+str(corr_support)+'-open-close-'+str(morph_support)+'-skip-'+str(skip_frames)+'-nlevs-'+str(nlevs)+'-thresh-'+str(thresh)+'-confidence-'+str(confidence)+'.mp4'

#slides_dir=results_dir+start_name+"-slides"
#os.system("mkdir "+slides_dir)

cap = cv2.VideoCapture(in_name)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(out_name ,fourcc, fps, (frame_width,frame_height),True)


print (frame_width,frame_height)
tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#tot_time=5000
ret = True

#num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames=tot_time
print num_frames

f=open(results_dir+start_name+".csv",mode='w')
#f.write("0")
num_channels=3


old_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
new_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
mask_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
corr_frame=np.zeros((frame_height,frame_width),dtype=np.float)
bit_frame=np.zeros((frame_height,frame_width),dtype=np.bool)
bool_frame=np.zeros((frame_height,frame_width),dtype=np.bool)
bitmask=np.zeros((frame_height,frame_width),dtype=np.uint8)
out_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
t=0

#first frame
cap.set(cv2.CAP_PROP_POS_FRAMES,t);
ret,frame=cap.read()
pyr_blender=pyrMask.pyrMask(frame,True,nlevs,0)
old_frame[:,:,:]=frame
started=False
tle=0
print tot_time
t = t+1

while(cap.isOpened() and ret):
    print t
    cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*t);
    ret, frame = cap.read()
    new_frame[:,:,:]=frame
    tic=time.time()
    corr_frame[:,:]=corrConv.fastCorr(old_frame,new_frame,corr_support)
    print time.time()-tic
    bitmask[:,:]=np.uint8(corr_frame>confidence)*255
    bitmask[:,:]=cv2.morphologyEx(cv2.morphologyEx(bitmask,cv2.MORPH_CLOSE,np.ones((morph_support,morph_support))),cv2.MORPH_OPEN,np.ones((morph_support,morph_support)))
    #bitmask[:,:]=
    for color in range(num_channels):
        mask_frame[:,:,color]=bitmask
    cv2.imshow(str(t),(mask_frame*255)*new_frame)
    out_frame[:,:,:]=pyr_blender.maskOut(frame,mask_frame)
    tle=pyr_blender.get_top_level_energy()
    if not(started):
        bit_frame[:,:]=bit_frame|(bitmask==255)
        if np.product(bit_frame):
            started=True
            print "started"
            f.write(str(skip_frames*t)+','+str(tle)+'\n')
            out.write(out_frame)
    else:
        f.write(str(skip_frames*t)+','+str(tle)+'\n')
        out.write(out_frame)
    t=t+1
    old_frame[:,:,:]=new_frame
    if t%10==0:
        cv2.destroyAllWindows()
    if t*skip_frames >= tot_time:
        break




f.close()
cap.release()
cv2.destroyAllWindows()
