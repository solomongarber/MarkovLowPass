import numpy as np
import cv2
import pyrMask
from scipy import ndimage
import os



skip_frames=100
blur_width=100
results_dir='./avg_abs_diff/'

#in_name='../Calc2FirstOrderDiffEqSepofVars.mp4'
#start_name='calc_2'

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

in_name='../calc2-avg-sample.mp4'
start_name='calc-2-avg'
skip_frames=1

#in_name='../preProcessGPU.avi'
#start_name='pre-process'

#in_name='../non-euclid-subsamp-100.avi'
out_name=results_dir+start_name+'-abs-diff-blur-'+str(blur_width)+'-skip-'+str(skip_frames)

slides_dir=results_dir+start_name+"-slides"
os.system("mkdir "+slides_dir)

cap = cv2.VideoCapture(in_name)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(out_name +'.mp4',fourcc, fps, (frame_width,frame_height),True)


print (frame_width,frame_height)
tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#tot_time=5000
ret = True

#num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames=tot_time
print num_frames

f=open(out_name+".csv",mode='w')
f.write("0")
num_channels=3
perfect_score=num_channels*255

old_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.int16)
new_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.int16)
mask_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
last_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
diff_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.int16)
sum_frame=np.zeros((frame_height,frame_width),dtype=np.uint8)
bool_frame=np.zeros((frame_height,frame_width),dtype=np.bool)
bit_frame=np.zeros((frame_height,frame_width),dtype=np.bool)
erode_frame=np.zeros((frame_height,frame_width),dtype=np.float32)
dilate_frame=np.zeros((frame_height,frame_width),dtype=np.float32)
out_frame=np.zeros((frame_height,frame_width,num_channels),dtype=np.uint8)
time=0

#first frame
cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*time);
ret,frame=cap.read()
pyr_blender=pyrMask.pyrMask(frame,True,0,0)
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
    sum_frame[:,:]=np.sum(diff_frame,2)/3
    med=np.median(sum_frame)
    avg=np.mean(sum_frame)
    #threshold=int((med+avg)/2)
    threshold=med
    #threshold=avg
    thresh=cv2.threshold(ndimage.uniform_filter(sum_frame,blur_width),threshold,255,cv2.THRESH_BINARY_INV)[1]
    
    for color in range(num_channels):
        #mask_frame[:,:,color]=255*dilate_frame
        mask_frame[:,:,color]=thresh
    #cv2.imshow('key',mask_frame)
    cv2.imshow(str(time),(mask_frame*255)*frame)
    last_frame[:,:,:]=out_frame
    out_frame[:,:,:]=pyr_blender.maskOut(frame,mask_frame)
    new_tle=pyr_blender.get_top_level_energy()
    f.write(','+str(new_tle))
    if time%10==0:
        cv2.destroyAllWindows()
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
    if time*skip_frames >= tot_time:
        break



#out.write(outframe)
f.close()
cap.release()
cv2.destroyAllWindows()
