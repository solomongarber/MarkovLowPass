import cv2
import numpy as np
import pyrStripe
import lPyr
import signal
in_name='../subsamp-change-threshold-0.06-MH12non-euclid.avi'
cap = cv2.VideoCapture(in_name)
cap.set(cv2.CAP_PROP_POS_FRAMES,4)
ret, frame4 = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES,100)
ret, frame100 = cap.read()
n=pyrStripe.pyrStripe(frame4)
p=n.stripeOut(frame100,200,800)
#cv2.imshow('p',p)
#print 't'
def preent():
    print 't'

def iirr(signal,r):
    y=np.zeros(signal.shape[0],dtype=np.int64)
    z=signal[0]
    y[0]=z
    for n in range(1,signal.shape[0]):
        z=z*r+signal[n]*(1-r)
        y[n]=z
    return y

def rms(signal,support):
    return signal

def battle(signal,start):
    y=np.zeros(signal.shape[0],dtype=np.int64)
    z=signal[start]
    y[0]=z
    velocity=0
    position=z
    for n in range(start+1,signal.shape[0]):
        x=signal[n]
        delta=x-z
        velocity=velocity+delta
        z=x
        position=position+velocity
        y[n]=position
    return y

def battle_iir(signal,r,start):
    y=np.zeros(signal.shape[0],dtype=np.int64)
    z=signal[start]
    y[0]=z
    velocity=0
    position=z
    for n in range(start+1,signal.shape[0]):
        x=signal[n]
        delta=x-z
        velocity=velocity*r+delta*(1-r)
        z=x
        position=position+velocity
        y[n]=position
    return y

def adjust_to_max(signal,support,candidates):
    num_samples=signal.shape[0]
    ind=0
    new_answers=np.zeros(len(candidates))
    for candidate in candidates:
        left=max([candidate-support,0])
        right=min([candidate+support,num_samples])
        choices=signal[range(int(left),int(right))]
        new_ans=left+np.argmax(choices)
        new_answers[ind]=new_ans
        ind+=1
    return new_answers


def get_tle_vid(vid_name):
    cap = cv2.VideoCapture(vid_name)
    time = 0
    tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    energies=np.zeros(tot_time,dtype=np.int64)
    cap.set(cv2.CAP_PROP_POS_FRAMES,time)
    ret,frame = cap.read()
    in_frame=np.zeros(frame.shape,dtype=np.int64)
    small_frame=cv2.pyrDown(frame)
    blur_frame=np.zeros(frame.shape,dtype=np.int64)
    print tot_time
    while(time<tot_time):
        in_frame[:,:,:]=frame
        small_frame[:,:,:]=cv2.pyrDown(frame)
        blur_frame[:,:,:]=cv2.pyrUp(small_frame)[:frame.shape[0],:frame.shape[1],:]
        tle=np.sum(np.abs(np.subtract(in_frame,blur_frame)))
        energies[time]=tle
        if time%50==0:
            print time
        time=time+1
        cap.set(cv2.CAP_PROP_POS_FRAMES,time)
        ret,frame = cap.read()
        
    return energies
        
