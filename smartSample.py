import cv2
import numpy as np
def sample(in_name,out_name,big_skip,small_skip):
    cap = cv2.VideoCapture(in_name)
    sum=0
    ret,frame=cap.read()
    tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print tot_time
    time=1
    old_frame=np.zeros(frame.shape,dtype=np.int64)
    old_frame[:,:,:]=frame
    new_frame=np.zeros(frame.shape,dtype=np.int64)
    diff_frame=np.zeros(frame.shape,dtype=np.int64)
    while time*big_skip<tot_time:
        cap.set(cv2.CAP_PROP_POS_FRAMES,big_skip*time)
        ret,frame=cap.read()
        new_frame[:,:,:]=frame
        sum+=np.sum(np.abs(np.subtract(old_frame,new_frame)))
        time+=1
        old_frame[:,:,:]=new_frame
        if time%100==1:
            print time
    avg=sum/time

    fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_name ,fourcc, fps, (frame_width,frame_height),True)

    time=0
    tot=0
    while time*small_skip<tot_time:
        cap.set(cv2.CAP_PROP_POS_FRAMES,small_skip*time)
        ret,frame=cap.read()
        new_frame[:,:,:]=frame
        sum=np.sum(np.abs(np.subtract(new_frame,old_frame)))
        if sum>avg:
            out.write(frame)
            old_frame[:,:,:]=new_frame
            tot+=1
            print tot
        time+=1
        if time%100==1:
            print time
    
    
        
