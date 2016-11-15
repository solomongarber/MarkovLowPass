import cv2
from collections import deque
#pycuda boilerplate
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import time
height = 10
width = 10

frameTotal = gpuarray.to_gpu(np.zeros((height,width, 3), dtype=np.uint32))
frame = gpuarray.to_gpu(np.ones((10,10,3)))

tot = frameTotal + frame
for i in range(255):
    tot = tot + frame

print tot
