import numpy as np
from collections import deque
import lPyr

class dataPyrQueue:
    def __init__(self,input_shape,frame_shape,small_shape,divisor,num_pixels,num_pixels_big,nD,smooth_exp,smooth_mult,data_exp,data_mult,data_local_exp,data_local_mult,stack_em):
        self.input_shape=input_shape
        self.frame_shape=frame_shape
        self.small_square_shape=(input_shape[0]/divisor,input_shape[1]/divisor,3)
        self.small_shape=small_shape
        self.divisor=divisor
        self.bigvisor=divisor*divisor
        self.nD=nD
        self.num_pixels_big=num_pixels_big
        self.num_pixels=num_pixels
        self.smooth_exp=smooth_exp
        self.smooth_mult=smooth_mult
        self.data_local_exp=data_local_exp
        self.data_local_mult=data_local_mult
        self.data_mult=data_mult
        self.data_exp=data_exp
        self.middle=nD/2+1
        self.frames = deque()
        self.big_frames=deque()
        self.stack_em=stack_em
        for i in range(nD+2):
            self.big_frames.append(np.zeros(frame_shape,dtype=np.uint8))
            if stack_em:
                self.frames.append(np.zeros((small_shape[0],small_shape[1],self.bigvisor),dtype=np.int)+255*(i%2))
            else:
                self.frames.append(np.zeros((small_shape[0],small_shape[1]),dtype=np.int)+255*(i%2))
        self.now=np.array(self.big_frames[self.middle],dtype=np.int)
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
        if self.stack_em:
            ans=np.zeros((self.small_square_shape[1],self.small_square_shape[0],3,self.bigvisor),dtype=np.int)
        else:
            ans=np.zeros((self.small_square_shape[1],self.small_square_shape[0],3),dtype=np.int)
        x=self.input_shape[1]
        y=self.input_shape[0]
        for i in range(self.divisor):
            for j in range(self.divisor):
                if self.stack_em:
                    ans[:,:,:,self.divisor*i+j]+=frame[0+i:x:self.divisor,0+i:y:self.divisor,:]
                else:
                    ans[:,:,:]+=frame[0+i:x:self.divisor,0+i:y:self.divisor,:]
        if self.stack_em:
            return np.reshape(ans,(self.small_square_shape[0]*self.small_square_shape[1],3,self.bigvisor))
        else:
            return np.reshape(ans,(self.small_square_shape[0]*self.small_square_shape[1],3))/self.bigvisor

    def add_frame(self, frame):
        small_frame=self.down_vec(frame)
        big_frame=np.zeros((self.num_pixels_big,3),dtype=np.uint8)
        big_frame[:,:]=np.reshape(frame,big_frame.shape)
        self.frames.append(small_frame)
        self.big_frames.append(big_frame)
        self.used.append(self.big_frames.popleft())
        self.frames.popleft()
        self.now=self.big_frames[self.middle]

    def add_frame_left(self):
        self.big_frames.appendleft(self.used.pop())
        self.big_frames.pop()
        self.now=self.big_frames[self.middle]

    def get_frame(self, label):
        return self.frames[label]

    def get_dcost(self,label):
        #print self.now.shape
        #print self.frames[label+1].shape
        #return 10*np.sum(np.abs(np.array(self.now,dtype=np.int)-np.array(self.frames[label+1],dtype=np.int)),1)+10self.get_local(label)
        return self.data_mult*self.get_exp_diffs(self.middle,label+1,self.data_exp)+self.get_local(label)

    def get_local(self,label):
        #back_energy=10*np.sum(np.power(np.abs(np.array(self.frames[label+1],dtype=np.int)-np.array(self.frames[label],dtype=np.int),self.),1)
        #front_energy=10*np.sum(np.power(np.abs(np.array(self.frames[label+1],dtype=np.int)-np.array(self.frames[label+2],dtype=np.int),sle),1)
        return self.data_local_mult*(self.get_exp_diffs(label+1,label,self.data_local_exp)+self.get_exp_diffs(label+1,label+2,self.data_local_exp))
    
    def get_exp_diffs(self,l1,l2,pow):
        if self.stack_em:
            return np.sum(np.sum(np.power(np.abs(np.array(self.frames[l1],dtype=np.int)-np.array(self.frames[l2],dtype=np.int)),pow),1),1)
        else:
            return np.sum(np.power(np.abs(np.array(self.frames[l1],dtype=np.int)-np.array(self.frames[l2],dtype=np.int)),pow),1)

    def get_scost(self,frame):
        smenergy=np.zeros((self.num_pixels,self.nD),dtype=np.int)
        for l in range(self.nD):
            if self.stack_em:
                smenergy[:,l]=np.sum(np.sum(np.power(np.abs(np.array(self.frames[l],dtype=np.int)-np.array(frame,dtype=np.int)),self.smooth_exp),1),1)
            else:
                smenergy[:,l]=np.sum(np.power(np.abs(np.array(self.frames[l],dtype=np.int)-np.array(frame,dtype=np.int)),self.smooth_exp),1)
        return self.smooth_mult*smenergy

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



