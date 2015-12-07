import cv2
from collections import deque
#pycuda boilerplate
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import time
start = time.time()
in_name='timelapseVideo.avi'
cap = cv2.VideoCapture(in_name)

use_gpu = True
out_name = 'pycudaMeanGPU='+str(use_gpu)+'.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#out = cv2.VideoWriter(out_name ,fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (fheight,fwidth),True)

out = cv2.VideoWriter(out_name ,fourcc, 30, (width,height),True)
frameList = deque()
ret = True

if(use_gpu):
    frameTotal = gpuarray.to_gpu(np.zeros((height,width, 3), dtype=np.uint32))
else:
    frameTotal = np.zeros((height,width,3), dtype=np.uint32)
#gpu_frameTotal = gpuarray.to_gpu(np.zeros((height,width, 3), dtype=np.uint32))
print cap.read()
while(cap.isOpened() and ret):
    ret, frame = cap.read()
    if(ret):
        if(use_gpu):
            frame = gpuarray.to_gpu(frame.astype(np.float32))


        #gpu_frame = gpuarray.to_gpu(frame.astype(np.uint32))
        #gpu_frameTotal = gpu_frameTotal + gpu_frame
        frameTotal = frameTotal + frame
        #frameList.append(gpu_frame)
        frameList.append(frame)
        if(len(frameList) > 10):
            frameRemoved = frameList.popleft()

            frameTotal = frameTotal - frameRemoved
        #print np.amax(frameTotal)
        # if(use_gpu):
        #     gpu_frameTotal = gpuarray.to_gpu(frameTotal.astype(np.float32))
        #     outFrame = (gpu_frameTotal/len(frameList)).get()
        # else:

        outFrame = frameTotal/len(frameList)
        #print outFrame
        if use_gpu:
            outFrame = outFrame.get()
        out.write(outFrame.astype(np.uint8))

print time.time() - start
#a_gpu = gpuarray.to_gpu(np.ones((2,4), dtype=np.float32))
#a_doubled = (2 * a_gpu).get()
#print a_gpu
#print a_doubled
