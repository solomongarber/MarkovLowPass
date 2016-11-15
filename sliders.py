import numpy as np
import cv2
import corrConv
import pyrTest

nums=[77,199,321,369,463,581,684,794,860]
old=cv2.pyrDown(cv2.imread("./correlate/slidetest/fs4.png"))
otle=pyrTest.abs_tle_im(old)
resultsdir='./slide_debug/'

for num in nums:
    new=cv2.pyrDown(cv2.imread("./correlate/slidetest/fs"+str(num)+".png"))
    corrpic=corrConv.fastCorr(old,new,13)
    cp=np.dstack((corrpic,corrpic,corrpic))
    print cp.shape
    ntle=pyrTest.abs_tle_im(new)
    info_kept=np.multiply(otle,np.float64(cp>0.8))
    p=pyrTest.rep_3d_name(np.subtract(otle,info_kept),'ilost'+str(num))
    cv2.imwrite(resultsdir+'ilost'+str(num)+'.png',p)
    info_lost=np.sum(np.subtract(otle,info_kept))
    print info_lost
    info_retained=np.multiply(ntle,np.float64(cp>0.8))
    info_gained=np.sum(np.subtract(ntle,info_retained))
    p=pyrTest.rep_3d_name(np.subtract(ntle,info_retained),'igained'+str(num))
    cv2.imwrite(resultsdir+'igot'+str(num)+'.png',p)
    print info_gained
    if info_gained<info_lost:
        print "better "+str(num)
    

    
    old[:,:,:]=new
    otle[:,:,:]=ntle
