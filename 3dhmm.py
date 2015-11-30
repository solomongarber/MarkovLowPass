import math
import numpy as np
import cv2
import dataQueue
import tables as tb
#from moviepy.editor import VideoFileClip
distance = 9
data_mult = 1
smooth_exp = 2
local_energy = False
data_enery_exp = 1
distance_decay=.9
smooth_lasso_exp=2
smooth_local_energy = True
smooth_local_exp = -1
smooth_local_mult=200
data_local_mult=1
data_mult=2
smooth_mult=1
data_exp = 1
#in_name='shrunken-subsamp-ne-100.avi'
in_name='timelapseVideo.avi'
#in_name='vitoybi.avi'

#clip=VideoFileClip(in_name)
out_name='used1dhmm-dist-'+str(distance)+'-data_exp-'+str(data_exp)+'-smooth_exp-'+str(smooth_exp)+'.avi'#'.mp4'
cap = cv2.VideoCapture(in_name)
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fwidth=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fheight=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#out = cv2.VideoWriter(out_name ,fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (fheight,fwidth),True)
fps=cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(out_name ,fourcc, 30, (fwidth,fheight),True)
print (fwidth,fheight)
#tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
tot_time=20
ret = True

#num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames=tot_time
print num_frames
nD = 2*distance+1
num_pixels=fwidth*fheight
num_channels=3
curr_data=dataQueue.dataQueue((fwidth,fheight,3),(num_pixels,num_channels),num_pixels,nD)

old_energy=np.zeros((num_pixels,nD),dtype=np.int)
new_energy=np.zeros((num_pixels,nD),dtype=np.int)
next_energy=np.zeros(old_energy.shape,dtype=np.int)
labels=np.zeros((num_pixels,nD),dtype=np.uint8)
denergy=np.zeros(num_pixels,dtype=np.int)
smenergy=np.zeros((num_pixels,nD),dtype=np.int)
curr_frame=np.zeros((num_pixels,num_channels),dtype=np.uint8)
time=0

#f=tb.openFile("labels.h5",'w')
#f.createCArray(f.root,'backpointers',tb.UInt8Atom(),(num_pixels,nD,num_frames))
#bp=f.root.backpointers
bp=np.zeros((num_pixels,nD,num_frames),dtype=np.uint8)
#vid=np.zeros((num_pixels,num_channels,num_frames),dtype=np.uint8)
while(cap.isOpened() and ret):
    print time
    ret, frame = cap.read()
    #frame=np.zeros(frame.shape,dtype=np.uint8)
    #if time == 0:
    #    frame[:,2,:]=255
    #    frame[2,:,:]=255
    #frame=clip.get_frame(time/fps)
    curr_frame=np.reshape(frame,(num_pixels,num_channels))
    #vid[:,:,time]=curr_frame
    curr_data.add_frame(curr_frame)
    #curr_frame=curr_data.get_curr_frame()
    for label in range(nD):
        label_frame=curr_data.get_frame(label+1)
        denergy[:]=curr_data.get_dcost(label)
        smenergy[:,:]=curr_data.get_scost(label_frame)
        new_energy[:,:]=old_energy+smenergy
        #new_energy=smenergy
        labels[:,label]=np.array(np.argmin(new_energy,1),dtype=np.uint8)
        #labels[:,label]=np.array(np.argmax(new_energy,1),dtype=np.uint8)
        next_energy[:,label]=new_energy[range(num_pixels),labels[:,label]]+denergy
        #next_energy[:,label]=new_energy[range(num_pixels),labels[:,label]]
    bp[:,:,time]=labels
    #print labels
    #bp.flush()
    old_energy[:,:]=next_energy[:,:]
    time=time+1
    if time == tot_time:
        break

#time=time-nD-3
time=time-nD-2
final_label=np.array(np.argmin(old_energy,1),dtype=np.uint8)
prev_labels=np.zeros(num_pixels,dtype=np.uint8)
prev_labels[:]=labels[range(num_pixels),final_label]
#outframe=cv2.cvtColor(curr_data.get_output(final_label),3)
outframe=np.zeros(frame.shape,dtype=np.uint8)
outframe[:,:,:]=curr_data.get_output(final_label)
#print outframe
out.write(outframe)
for t in range(time,-1,-1):
    labels[:,:]=bp[:,:,t+nD]
    #print labels
    #cap.set(cv2.CAP_PROP_POS_FRAMES,1000*t/fps);
    #cap.set(cv2.CAP_PROP_POS_FRAMES,t);
    #ret,frame=cap.read()
    #print frame.shape
    #frame=clip.get_frame(t/fps)
    #curr_frame[:,:]=np.reshape(frame,(num_pixels,num_channels))
    #curr_frame[:,:]=vid[:,:,t-1]
    #curr_frame=np.zeros((num_pixels,num_channels),dtype=np.uint8)
    #curr_data.add_frame_left(curr_frame)
    curr_data.add_frame_left()
    outframe[:,:,:]=curr_data.get_output(prev_labels)
    out.write(outframe)
    #print outframe
    prev_labels[:]=labels[range(num_pixels),prev_labels]
    print t




#f.close()
cap.release()
cv2.destroyAllWindows()
