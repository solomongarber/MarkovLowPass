import cv2
in_name='originalVideo.mp4'
#in_name='timelapseVideo.avi'
cap = cv2.VideoCapture(in_name)

out_name = 'preProcessSimple.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tot_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
out = cv2.VideoWriter(out_name ,fourcc, 30, (width,height),True)
current_frame_num = 0
skip_by = 100
ret = True

while(cap.isOpened() and ret):
    ret, frame = cap.read()
    if(ret):
        print "on frame: {} out of: {}".format(current_frame_num,tot_num_frames)
        out.write(frame)
        current_frame_num = current_frame_num + skip_by
        cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame_num)
