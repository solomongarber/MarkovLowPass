import numpy as np
import cv2

skip_frames=4
start_frame=5
in_name='./med_pyr_maxmin/debugcalc_2-pyr-maxmin-block-delete-2.5-med-support1-skip-100-loc-mul-0.5-loc-exp2-twolevs-divisor-16-1dhmm-dist-20-data_mult-1-data-exp1smooth-mult5-smooth_exp-2-stack_emFalse.mp4'

out_name='calc2-fingersoup.mp4'
print "easy"
cap = cv2.VideoCapture(in_name)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
big_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

big_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps=cap.get(cv2.CAP_PROP_FPS)
print "medium"
out = cv2.VideoWriter(out_name ,fourcc, fps, (big_width,big_height),True)

tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print "hard"
ret=True
time=0
while(cap.isOpened() and ret):
    print time
    cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*time+start_frame);
    ret, frame = cap.read()
    time=time+1
    if ret:
        out.write(frame)
    if time*skip_frames +start_frame>= tot_time:
        break
