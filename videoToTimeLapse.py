import numpy as np
import cv2

cap = cv2.VideoCapture('shrunken-subsamp-ne-100.avi')
multiplier = 5
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('timelapseVideo.mp4',fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

ret = True
count = int(cap.get(cv2.CAP_PROP_FPS))
print "creating timelapse..."
frames = 1
while(cap.isOpened() and ret == True):
    ret, frame = cap.read()

    frames+= 1
    if ret == True and count % int(cap.get(cv2.CAP_PROP_FPS) * multiplier) == 0:
        out.write(frame)
        count = 0
        print "printing frame #: {}".format(frames)
        #print "added frame"
    count += 1


    #cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
#cv2.destroyAllWindows()
print "done!"
