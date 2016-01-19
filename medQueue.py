import numpy as np

class medQueue:
    def __init__(self,support,frame_shape,num_pixels,num_channels):
        self.frames=np.zeros((num_pixels,num_channels,support),dtype=np.uint8)
        for i in range(support/2):
            self.frames[:,:,i*2]=255
        self.frame_shape=frame_shape
        self.ind=0
        self.support=support
        self.num_pixels=num_pixels
        self.num_channels=num_channels


    def add_frame(self,frame):
        self.frames[:,:,self.ind]=np.reshape(frame,(self.num_pixels,self.num_channels))
        self.ind=np.mod(self.ind+1,self.support)

    def get_median(self):
        ans=np.median(self.frames,2)
        return np.reshape(ans,self.frame_shape)
