import numpy as np
import cv2
from lukaQueue2 import lukaQueue
import time


def meanFrame(frame, queue):
    return queue.getAverageFrame()
#    for lineIndex, line in enumerate(frame):
#        for pixelIndex, pixel in enumerate(line):
#            for pixelValueIndex, pixelValue in enumerate(pixel):
#                frame[lineIndex][pixelIndex][pixelValueIndex] = queue.averageOf(lineIndex, pixelIndex, pixelValueIndex)
#    return frame


begin = time.time()
cap = cv2.VideoCapture('timelapseVideo.avi')
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('timelapseVideo3.avi',fourcc, cap.get(cv2.cv.CV_CAP_PROP_FPS), (1280,720))
ret = True
count = 0
queue = lukaQueue()
while(cap.isOpened() and ret == True):
    ret, frame = cap.read()
    if ret == True:
        count += 1
        print "On frame number: {}".format(count)
        #clonedFrame = frame

        queue.addFrame(frame)
        newFrame = meanFrame(frame, queue)

        out.write(newFrame)
        #print "added frame"



cap.release()
#cv2.destroyAllWindows()
end = time.time()
print "done!"
print "it took: {} seconds to finish!".format(end-begin)
