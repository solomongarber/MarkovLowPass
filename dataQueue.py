import numpy as np
from collections import deque

class dataQueue:
    def __init__(self,frame_shape,size,num_pixels,nD):
        self.frame_shape=frame_shape
        self.size=size
        self.nD=nD
        self.num_pixels=num_pixels
        self.middle=nD/2+1
        self.frames = deque()
        for i in range(nD+1):
            self.frames.append(np.zeros(size,dtype=np.uint8))
        self.now=np.array(self.frames[self.middle],dtype=np.int)
            


    def add_frame(self, frame):
        self.frames.append(frame)
        self.frames.popleft()
        self.now=self.frames[self.middle]

    def add_frame_left(self,frame):
        self.frames.appendleft(frame)
        self.frames.pop()
        self.now=self.frames[self.middle]

    def get_frame(self, label):
        return self.frames[label]

    def get_dcost(self,label):
        return .1* np.sum(np.abs(np.array(self.now,dtype=np.int)-np.array(self.frames[label+1]),dtype=np.int),1)

    def get_scost(self,frame):
        smenergy=np.zeros((self.num_pixels,self.nD),dtype=np.int)
        for l in range(self.nD):
            smenergy[:,l]=np.sum(np.power(np.abs(np.array(self.frames[l],dtype=np.int)-np.array(frame,dtype=np.int)),3),1)
        return smenergy

    def get_output(self,labels):
        output=np.zeros((self.frame_shape[1],self.frame_shape[0],3),dtype=np.uint8)
        bad_idea=np.zeros((self.num_pixels,3,self.nD),dtype=np.uint8)
        for l in range(self.nD):
            bad_idea[:,:,l]=self.get_frame(l+1)
        for color in range(3):
            output[:,:,color]=np.reshape(bad_idea[range(self.num_pixels),color,self.nD-labels-1],(self.frame_shape[1],self.frame_shape[0]))
            #output[:,:,color]=np.reshape(10*labels,(self.frame_shape[1],self.frame_shape[0]))
        return output



