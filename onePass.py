#import math
import numpy as np
import cv2
import pyrMask
from scipy import ndimage
import os

erode_support=5
dilate_support=44 + erode_support
epsilon=1./(dilate_support*dilate_support*2)
skip_frames=100
#skip_ahead=0

results_dir='./erode_dilate/'

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
#skip_ahead=6

#in_name='../calc2-avg-sample.mp4'
#start_name='calc-2-avg'
#skip_frames=1

#in_name='../preProcessGPU.avi'
#start_name='pre-process'

#in_name='../non-euclid-subsamp-100.avi'
out_name=results_dir+start_name+'-erode-'+str(erode_support)+'-dilate-'+str(dilate_support)+'-skip-'+str(skip_frames)+'-ahead-'+str(skip_ahead)+'.mp4'

slides_dir=results_dir+start_name+"-slides"
os.system("mkdir "+slides_dir)

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
f.write("0")
num_channels=3
perfect_score=num_channels*255

old_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.int16)
new_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.int16)
mask_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
last_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
diff_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.int16)
vote_frame=np.zeros((frame_height,frame_width),dtype=np.float32)
bool_frame=np.zeros((frame_height,frame_width),dtype=np.bool)
bit_frame=np.zeros((frame_height,frame_width),dtype=np.bool)
erode_frame=np.zeros((frame_height,frame_width),dtype=np.float32)
dilate_frame=np.zeros((frame_height,frame_width),dtype=np.float32)
out_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
time=skip_ahead

#first frame
cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*time);
ret,frame=cap.read()
pyr_blender=pyrMask.pyrMask(frame,True)
old_frame[:,:,:]=frame
started=False
up=True
old_tle=pyr_blender.get_top_level_energy()
new_tle=0
print old_tle
time = time+1
best_score=255*num_channels
while(cap.isOpened() and ret):
    print time
    cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*time);
    ret, frame = cap.read()
    new_frame[:,:,:]=frame
    diff_frame[:,:,:]=np.abs(old_frame-new_frame)
    med=np.median(diff_frame)
    avg=np.mean(diff_frame)
    #threshold=int((med+avg)/2)
    threshold=med
    thresh=cv2.threshold(diff_frame,threshold,255,cv2.THRESH_BINARY)
    vote_frame[:,:]=np.sum(thresh[1],2)
    bool_frame[:,:]=vote_frame==best_score
    vote_frame[:,:]=bool_frame*1.0
    erode_frame[:,:]=ndimage.filters.uniform_filter(vote_frame,erode_support,mode='constant')
    bool_frame[:,:]=erode_frame==1
    erode_frame[:,:]=bool_frame*1.0
    dilate_frame[:,:]=ndimage.filters.uniform_filter(erode_frame,dilate_support,mode='constant')
    bool_frame[:,:]=dilate_frame>epsilon
    dilate_frame[:,:]=1-bool_frame
    for color in range(num_channels):
        mask_frame[:,:,color]=255*dilate_frame
    #cv2.imshow('key',mask_frame)
    #cv2.waitKey()
    last_frame[:,:,:]=out_frame
    out_frame[:,:,:]=pyr_blender.maskOut(frame,mask_frame)
    new_tle=pyr_blender.get_top_level_energy()
    f.write(','+str(new_tle))
    #if ((new_tle<old_tle)& up & started):
        #cv2.imwrite(slides_dir+"/frame"+str(time*skip_frames)+".jpg",last_frame)
    if (new_tle<old_tle):
        up=False
    else:
        up=True
    if not(started):
        bit_frame[:,:]=bit_frame|(~bool_frame)
        if np.product(bit_frame):
            started=True
            print "started"
            out.write(out_frame)
    else:
        out.write(out_frame)
    time=time+1
    old_frame[:,:,:]=new_frame
    old_tle=new_tle
    if time*skip_frames >= tot_time:
        break



#out.write(outframe)
f.close()
cap.release()
cv2.destroyAllWindows()
