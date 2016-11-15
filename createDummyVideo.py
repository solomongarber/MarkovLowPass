import numpy as np
import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('dummyVid.avi',fourcc, 30, (1280,720))

for i in range(0,150):
    frame = np.zeros((720,1280,3), dtype=np.uint8)
    if(60 < i < 70):
        frame[:,:,:] = 255
    out.write(frame)

print "done"
out.release()
