import cv2
#from collections import deque
#pycuda boilerplate
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import time
start = time.time()
in_name='originalVideo.mp4'
#in_name='timelapseVideo.avi'
cap = cv2.VideoCapture(in_name)

out_name = 'preProcessGPU.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tot_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
out = cv2.VideoWriter(out_name ,fourcc, 30, (width,height),True)
#variables for difference stuff
current_frame_num = 0
find_average_skip_by = 100
find_frames_skip_by = 10
difference_list = []
difference_multiplier = 2

#initial set up of prevframe
ret, prevframe = cap.read()
prevframe = gpuarray.to_gpu(prevframe.astype(np.float32))
#First, take a frame every 100, find average change per pixel of this video
while(cap.isOpened() and ret):
    ret, frame = cap.read()
    if(ret):
        frame = gpuarray.to_gpu(frame.astype(np.float32))
        difference_list.append(gpuarray.sum(abs(frame-prevframe)))
        prevframe = frame
        current_frame_num = current_frame_num + find_average_skip_by
        cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame_num)
        print "{}%% done".format((current_frame_num/tot_num_frames) * 50)
average_frame_diff = sum(difference_list)/len(difference_list)
average_frame_diff= average_frame_diff.get()


#Second, take a frame from every 10, and if it is greater than some const * average add it
#start back at 0
current_frame_num = 0
cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame_num)
#initial set up of prevframe
ret, prevframe = cap.read()
initial_frame = gpuarray.to_gpu(prevframe.astype(np.float32))

while(cap.isOpened() and ret):
    ret, frame = cap.read()
    if(ret):
        frame = gpuarray.to_gpu(frame.astype(np.float32))
        diff = (gpuarray.sum(abs(frame-initial_frame))).get()
        if(diff > int(average_frame_diff) * difference_multiplier):
            out.write(frame.get().astype(np.uint8))
            initial_frame = frame
            print "wrote out"
        current_frame_num = current_frame_num + find_frames_skip_by
        cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame_num)
        print "{}%% done".format(((current_frame_num/tot_num_frames) * 50) + 50)



print "time taken was: {}".format(time.time()-start)
