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


def get_tle_vid_goos(vid_name,divisor):
    cap = cv2.VideoCapture(vid_name)
    time = 0
    tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    energies=np.zeros(tot_time,dtype=np.int64)
    cap.set(cv2.CAP_PROP_POS_FRAMES,time)
    ret,frame = cap.read()
    in_frame=np.zeros(frame.shape,dtype=np.int64)
    small_frame=cv2.pyrDown(frame)
    blur_frame=np.zeros(frame.shape,dtype=np.int64)
    goose=cv2.getGaussianKernel(frame.shape[0],frame.shape[0]/divisor)
    goose1=cv2.getGaussianKernel(frame.shape[1],frame.shape[1]/divisor)
    goosq=np.dot(goose,np.transpose(goose1))
    m=1/np.max(goosq)
    center_frame=np.zeros(frame.shape,dtype=np.float64)
    for color in range(3):
        center_frame[:,:,color]=goosq*m
    print tot_time
    while(time<tot_time):
        in_frame[:,:,:]=frame
        small_frame[:,:,:]=cv2.pyrDown(frame)
        blur_frame[:,:,:]=cv2.pyrUp(small_frame)[:frame.shape[0],:frame.shape[1],:]
        tle=np.sum(center_frame*np.abs(np.subtract(in_frame,blur_frame)))
        energies[time]=tle
        if time%50==0:
            print time
        time=time+1
        cap.set(cv2.CAP_PROP_POS_FRAMES,time)
        ret,frame = cap.read()
        
    return energies
        
def get_sq_tle_vid(vid_name):
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
        tle=np.sum(np.square(np.abs(np.subtract(in_frame,blur_frame))))
        energies[time]=tle
        if time%50==0:
            print time
        time=time+1
        cap.set(cv2.CAP_PROP_POS_FRAMES,time)
        ret,frame = cap.read()
        
    return energies
def get_sq_tle_vid(vid_name):
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
        tle=np.sum(np.square(np.abs(np.subtract(in_frame,blur_frame))))
        energies[time]=tle
        if time%50==0:
            print time
        time=time+1
        cap.set(cv2.CAP_PROP_POS_FRAMES,time)
        ret,frame = cap.read()
        
    return energies

def get_sq_sle_vid(vid_name):
    cap = cv2.VideoCapture(vid_name)
    time = 0
    tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    energies=np.zeros(tot_time,dtype=np.int64)
    cap.set(cv2.CAP_PROP_POS_FRAMES,time)
    ret,frame = cap.read()
    in_frame=np.zeros(cv2.pyrDown(frame).shape,dtype=np.int64)
    #small_uint=cv2.pyrDown(frame)
    small_frame=np.zeros(cv2.pyrDown(cv2.pyrDown(frame)).shape,dtype=np.uint8)
    blur_frame=np.zeros(in_frame.shape,dtype=np.int64)
    print tot_time
    while(time<tot_time):
        in_frame[:,:,:]=cv2.pyrDown(frame)
        small_frame[:,:,:]=cv2.pyrDown(cv2.pyrDown(frame))
        blur_frame[:,:,:]=cv2.pyrUp(small_frame)[:blur_frame.shape[0],:blur_frame.shape[1],:]
        tle=np.sum(np.square(np.abs(np.subtract(in_frame,blur_frame))))
        energies[time]=tle
        if time%50==0:
            print time
        time=time+1
        cap.set(cv2.CAP_PROP_POS_FRAMES,time)
        ret,frame = cap.read()
    return energies

def vidvol(in_name,outshape,slope,numframes,lev,start_frame,r,skip,margin):
    time=1
    cap=cv2.VideoCapture(in_name)
    ans=np.zeros(outshape,dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    ret,frame=cap.read()
    for l in range(lev):
        frame=cv2.pyrDown(frame)
    y=frame.shape[0]
    x=frame.shape[1]
    ans[:y,margin:x+margin,3]=255
    ans[:y,margin:x+margin,:3]=frame
    framebox=np.zeros((y,x,4),dtype=np.uint8)
    framebox[:,:,3]=255
    while time<numframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame+time*skip)
        ret,frame=cap.read()
        for l in range(lev):
            frame=cv2.pyrDown(frame)
        inset=time/slope
        if np.random.rand(1)[0]>.9:
            frame[:,0,:]=0
            frame[:,0,2]=255
            frame[0,:,:]=0
            frame[0,:,2]=255
        framebox[:,:,:3]=frame
        t=time+margin
        ans[inset:y+inset,t:x+t,:]=r*ans[inset:y+inset,t:x+t,:]+(1-r)*framebox[:y,:x,:]
        time=time+1
    ans[inset:y+inset,time:x+time,:]=framebox
    return ans

def boxout(im,x,y,d):
    ans=np.zeros(im.shape,dtype=np.uint8)
    ans[:,:,:]=im
    ans[y-d:y+d+1,x-d,:]=0
    ans[y-d:y+d+1,x-d,2:]=255
    ans[y-d:y+d+1,x+d,:]=0
    ans[y-d:y+d+1,x+d,2:]=255
    ans[y-d,x-d:x+d+1,:]=0
    ans[y-d,x-d:x+d+1,2:]=255
    ans[y+d,x-d:x+d+1,:]=0
    ans[y+d,x-d:x+d+1,2:]=255
    return ans

def lineify(ans,start,end,num):
    inc=1.0/num
    for n in range(num):
        x=int((1-inc*n)*start[0]+inc*n*end[0])
        y=int((1-inc*n)*start[1]+inc*n*end[1])
        ans[x,y,:2]=0
        ans[x,y,2:]=255
    return ans



def mk_blowup(im1,im2,outshape,X,Y,delta,lev,displacement,margin):
    temp=np.zeros((im1.shape[0],im1.shape[1],2),dtype=np.int32)
    temp[:,:,0]=np.sum(im1,2)/3
    temp[:,:,1]=np.sum(im2,2)/3
    for color in range(3):
        im1[:,:,color]=np.uint8(temp[:,:,0])
        im2[:,:,color]=np.uint8(temp[:,:,1])
        
    sm1=np.zeros(im1.shape,dtype=np.uint8)
    sm1[:,:,:]=im1
    sm2=np.zeros(im1.shape,dtype=np.uint8)
    sm2[:,:,:]=im2
    for l in range(lev):
        sm1=cv2.pyrDown(sm1)
        sm2=cv2.pyrDown(sm2)
    scale=im1.shape[0]/sm1.shape[0]
    d=np.max((delta/scale,2))
    x=X/scale
    y=Y/scale
    print d
    sm1=boxout(sm1,y,x,d)
    sm2=boxout(sm2,y,x,d)
    ans=np.zeros(outshape,dtype=np.uint8)
    cursor=(sm1.shape[0],sm1.shape[1])
    ans[margin:cursor[0]+margin,margin:cursor[1]+margin,:3]=sm1
    ans[margin:cursor[0]+margin,margin:cursor[1]+margin,3]=255
    center=(margin+x,margin+y)
    end_center=(center[0]+displacement[0],center[1]+displacement[1])
    for combo in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        start=(center[0]+d*combo[0],center[1]+d*combo[1])
        end=(end_center[0]+delta*combo[0],end_center[1]+delta*combo[1])
        ans=lineify(ans,start,end,1000)
    clip=im1[X-delta:X+delta+1,Y-delta:Y+delta+1,:]
    intensities1=np.reshape(clip[:,:,0], np.product(clip[:,:,0].shape))
    
    ans[end_center[0]-delta:end_center[0]+delta+1,end_center[1]-delta:end_center[1]+delta+1,:3]=clip
    ans[end_center[0]-delta:end_center[0]+delta+1,end_center[1]-delta:end_center[1]+delta+1,3]=255
    ans=boxout(ans,end_center[1],end_center[0],delta)
    bcursor=(cursor[0],cursor[1]+sm1.shape[1]+margin)
    tcursor=(margin,sm1.shape[1]+2*margin)
    ans[tcursor[0]:bcursor[0]+margin,tcursor[1]:bcursor[1]+margin,:3]=sm2
    ans[tcursor[0]:bcursor[0]+margin,tcursor[1]:bcursor[1]+margin,3]=255
    center=(margin+x,2*margin+y+sm1.shape[1])
    end_center=(center[0]+displacement[0],center[1]+displacement[1])
    for combo in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        start=(center[0]+d*combo[0],center[1]+d*combo[1])
        end=(end_center[0]+delta*combo[0],end_center[1]+delta*combo[1])
        ans=lineify(ans,start,end,1000)
    clip=im2[X-delta:X+delta+1,Y-delta:Y+delta+1,:]
    intensities2=np.reshape(clip[:,:,0], np.product(clip[:,:,0].shape))
    ans[end_center[0]-delta:end_center[0]+delta+1,end_center[1]-delta:end_center[1]+delta+1,:3]=clip
    ans[end_center[0]-delta:end_center[0]+delta+1,end_center[1]-delta:end_center[1]+delta+1,3]=255
    ans=boxout(ans,end_center[1],end_center[0],delta)
    bcursor=(cursor[0],cursor[1]+sm1.shape[1]+margin)
    tcursor=(margin,sm1.shape[1]+2*margin)
    o=open('./correlate/results/x-'+str(X)+'-y-'+str(Y)+'-d-'+str(delta)+'.csv','w')
    for ind in range(len(intensities1)):
        o.write(str(intensities1[ind])+','+str(intensities2[ind])+'\n')
    o.close()
    cv2.imwrite('./correlate/results/x-'+str(X)+'-y-'+str(Y)+'-d-'+str(delta)+'.png',ans)
    return ans
