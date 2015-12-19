import numpy as np
import cv2
import lPyr
class pyrStripe:
    def __init__(self,im):
        self.pyr=lPyr.build_l_pyr(im);
        #while(not(min(im.shape)<10)):
        #    self.pyr=

    def stripeOut(self,im,r1,r2):
        new_pyr=lPyr.build_l_pyr(im);
        for band_num in range(len(new_pyr)):
            chunksz=np.power(2,band_num)
            first_break = r1/chunksz
            second_break=r2/chunksz
            first_lweight=(r1%chunksz)/float(chunksz)
            first_rweight=1-first_lweight
            second_lweight=(r2%chunksz)/float(chunksz)
            second_rweight=1-second_lweight
            old_band=self.pyr[band_num]
            new_band=new_pyr[band_num]
            if first_break>-1:
                old_band[:,:first_break,:]=new_band[:,:first_break,:]
                old_band[:,first_break,:]=first_lweight*new_band[:,first_break,:]+first_rweight*old_band[:,first_break,:]
            else:
                first_break=0
            if second_break<old_band.shape[1]-1:
                old_band[:,second_break+1:,:]=new_band[:,second_break+1:,:]
                old_band[:,second_break,:]=second_lweight*old_band[:,second_break,:]+second_rweight*new_band[:,second_break,:]
        return lPyr.recon_l_pyr(self.pyr)
