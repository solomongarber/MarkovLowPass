import cv2
import numpy as np
import time
start = time.time()
in_name='preProcessGPU.avi'
cap = cv2.VideoCapture(in_name)

ret = True
while(cap.isOpened() and ret):
    ret, frame = cap.read()
    if(ret):
        print "hello"
