import math
import numpy as np
import cv2
import dataQueue
import tables as tb
distance = 5
data_exp = 1
smooth_exp = 2
local_energy = False
data_enery_exp = 1
distance_decay=.9
smooth_reg_exp=2
smooth_local_energy = True
smooth_local_exp = -1
smooth_local_mult=100
data_local_mult=1
data_mult=2
smooth_mult=1
in_name='shrunken-subsamp-ne-100.avi'



out_name='1dhmm-dist-'+str(distance)+'-data_exp-'+str(data_exp)+'-smooth_exp-'+str(smooth_exp)+'.avi'
cap = cv2.VideoCapture(in_name)
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fwidth=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fheight=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_name ,-1, int(cap.get(cv2.CAP_PROP_FPS)), (fwidth,fheight))
print (fwidth,fheight)
#tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret = True
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print num_frames
nD = 2*distance+1
num_pixels=fwidth*fheight
num_channels=3
curr_data=dataQueue.dataQueue((fwidth,fheight,3),(num_pixels,num_channels),num_pixels,nD)

old_energy=np.zeros((num_pixels,nD),dtype=int)
new_energy=np.zeros((num_pixels,nD),dtype=int)
next_energy=np.zeros(old_energy.shape,dtype=int)
labels=np.zeros((num_pixels,nD),dtype=np.uint8)
denergy=np.zeros(num_pixels,dtype=int)
smenergy=np.zeros(num_pixels,dtype=int)
time=0

f=tb.openFile("labels.h5",'w')
f.createCArray(f.root,'backpointers',tb.UInt8Atom(),(num_pixels,nD,num_frames))
bp=f.root.backpointers


while(cap.isOpened() and ret):
    ret, frame = cap.read()
    curr_frame=np.reshape(frame,(num_pixels,num_channels))
    curr_data.add_frame(curr_frame)
    #curr_frame=curr_data.get_curr_frame()
    for label in range(nD):
        label_frame=curr_data.get_frame(label)
        denergy=curr_data.get_dcost(label)
        smenergy=curr_data.get_scost(label_frame)
        new_energy=old_energy+smenergy
        labels[:,label]=np.array(np.argmin(new_energy,1),dtype=np.uint8)
        next_energy[:,label]=new_energy[range(num_pixels),labels[:,label]]+denergy
    bp[:,:,time]=labels
    bp.flush()
    old_energy=next_energy
    time=time+1
    if time == 20:
        break
    print time

time=time-nD-2
final_label=np.uint8(np.argmin(old_energy,1))
prev_labels=labels[np.array(range(num_pixels)),final_label]
outframe=cv2.cvtColor(curr_data.get_output(final_label),3)
print outframe.dtype
out.write(outframe)
for t in range(time,-1,-1):
    labels=bp[:,:,t]
    cap.set(cv2.CAP_PROP_POS_FRAMES,t);
    ret,frame=cap.read()
    curr_frame=np.reshape(frame,(num_pixels,num_channels))
    curr_data.add_frame_left(curr_frame)
    outframe=cv2.fromarray(curr_data.get_output(prev_labels))
    out.write(outframe)
    prev_labels=labels[range(num_pixels),prev_labels]
    print t
    
    
    

f.close()
cap.release()


