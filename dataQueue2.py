import numpy as np
from collections import deque

class dataQueue2:
    def __init__(self,input_shape,frame_shape,small_shape,divisor,num_pixels,num_pixels_big,nD):
        self.input_shape=input_shape
        self.frame_shape=frame_shape
        self.small_square_shape=(input_shape[0]/divisor,input_shape[1]/divisor,3)
        self.small_shape=small_shape
        self.divisor=divisor
        self.bigvisor=divisor*divisor
        self.nD=nD
        self.num_pixels_big=num_pixels_big
        self.num_pixels=num_pixels
        self.middle=nD/2+1
        self.frames = deque()
        self.big_frames=deque()
        for i in range(nD+1):
            self.big_frames.append(np.zeros(frame_shape,dtype=np.uint8))
            self.frames.append(np.zeros(small_shape,dtype=np.int))
        self.now=np.array(self.frames[self.middle],dtype=np.int)
        self.used=deque()
        #x=input_shape[1]
        #frame=np.zeros((input_shape[0],input_shape[1]),dtype=np.int)
        #for i in range(input_shape[0]/divisor):
        #    frame[i*divisor:(i+1)*divisor,:]=np.array(range(x*i,x*(i+1)))/divisor
        #self.expand=np.reshape(frame,num_pixels_big)
        x=input_shape[0]
        frame=np.zeros((input_shape[1],input_shape[0]),dtype=np.int)
        for i in range(input_shape[1]/divisor):
            frame[i*divisor:(i+1)*divisor,:]=np.array(range(x*i,x*(i+1)))/divisor
        self.expand=np.reshape(frame,num_pixels_big)
        print self.expand
        
    def down_vec(self,frame):
        ans=np.zeros((self.small_square_shape[1],self.small_square_shape[0],3),dtype=np.int)
        x=self.input_shape[1]
        y=self.input_shape[0]
        s=frame[0:x:self.divisor,0:y:self.divisor,:]/self.bigvisor
        for i in range(self.divisor):
            for j in range(self.divisor):
                ans[:,:,:]+=frame[0+i:x:self.divisor,0+i:y:self.divisor,:]
        print
        return np.reshape(ans,(self.small_square_shape[0]*self.small_square_shape[1],3))/self.bigvisor
        

    def add_frame(self, frame):
        small_frame=self.down_vec(frame)
        big_frame=np.zeros((self.num_pixels_big,3),dtype=np.uint8)
        big_frame[:,:]=np.reshape(frame,big_frame.shape)
        self.frames.append(small_frame)
        self.big_frames.append(big_frame)
        self.used.append(self.big_frames.popleft())
        self.frames.popleft()
        self.now=self.frames[self.middle]

    def add_frame_left(self):
        self.big_frames.appendleft(self.used.pop())
        self.big_frames.pop()
        self.now=self.frames[self.middle]

    def get_frame(self, label):
        return self.frames[label]

    def get_dcost(self,label):
        #print self.now.shape
        #print self.frames[label+1].shape
        return 10*np.sum(np.abs(np.array(self.now,dtype=np.int)-np.array(self.frames[label+1],dtype=np.int),dtype=np.int),1)

    def get_scost(self,frame):
        smenergy=np.zeros((self.num_pixels,self.nD),dtype=np.int)
        for l in range(self.nD):
            smenergy[:,l]=np.sum(np.power(np.abs(np.array(self.frames[l],dtype=np.int)-np.array(frame,dtype=np.int)),2),1)
        return smenergy

    def get_frame_big(self,label):
        return self.big_frames[label]
        
    def get_output(self,labels):
        output=np.zeros((self.input_shape[1],self.input_shape[0],3),dtype=np.uint8)
        bad_idea=np.zeros((self.num_pixels_big,3,self.nD),dtype=np.uint8)
        for l in range(self.nD):
            bad_idea[:,:,l]=self.get_frame_big(l+1)
        for color in range(3):
            output[:,:,color]=np.reshape(bad_idea[range(self.num_pixels_big),color,labels[self.expand]],(self.input_shape[1],self.input_shape[0]))
        return output



