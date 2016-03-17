import numpy as np
from scipy.stats.stats import pearsonr
from scipy import ndimage

def correlate(pic1,pic2,support):
    p1_frame=np.pad(pic1,support,mode='reflect')
    p2_frame=np.pad(pic2,support,mode='reflect')
    dims=pic1.shape
    width=support*2+1
    r_vecs=np.zeros((pic1.shape[0],pic1.shape[1],width*width*2),dtype=np.float)
    ans=np.zeros((pic1.shape[0],pic1.shape[1]),dtype=np.float)
    for ind in range(width*width*2):
        if ind<width*width:
            x_off=ind%width
            y_off=ind/width
            r_vecs[:,:,ind]=p1_frame[x_off:dims[0]+x_off,y_off:dims[1]+y_off,2]
        else:
            oth_ind=(ind-width*width)
            x_off=oth_ind%width
            y_off=oth_ind/width
            r_vecs[:,:,ind]=p2_frame[x_off:dims[0]+x_off,y_off:dims[1]+y_off,2]
    ans[:,:]=[[pearsonr(y[:width*width],y[width*width:])[0] for y in x] for x in r_vecs]
    return ans
    #return r_vecs

def sequentialCorr(pic1,pic2,support):
    p1_frame=np.pad(pic1,support,mode='reflect')
    p2_frame=np.pad(pic2,support,mode='reflect')
    dims=pic1.shape
    width=support*2+1
    
    ans=np.zeros((pic1.shape[0],pic1.shape[1]),dtype=np.float)
    for ind in range(dims[0]*dims[1]):
        x=ind%dims[0]
        y=ind/dims[0]
        xbar=p1_frame[x-support:x+support,y-support:y+support,2]
        ybar=p2_frame[x-support:x+support,y-support:y+support,2]
        #print sum(xbar-ybar)
        #print pearsonr(np.reshape(xbar,np.product(xbar.shape)),np.reshape(ybar,np.product(ybar.shape)))[0]
        ans[x,y]=pearsonr(np.reshape(xbar,np.product(xbar.shape)),np.reshape(ybar,np.product(ybar.shape)))[0]
    return ans

def fastCorr(pic1,pic2,support):
    width=support*2+1
    numel=width*width
    p1=np.sum(np.array(pic1,dtype=np.float),2)/3
    p1_off=np.zeros(p1.shape,dtype=np.float)
    p2=np.sum(np.array(pic2,dtype=np.float),2)/3
    p2_off=np.zeros(p1.shape,dtype=np.float)
    means1=np.zeros(p1.shape,dtype=np.float)
    means2=np.zeros(p1.shape,dtype=np.float)
    sum_sq_diff1=np.zeros(p1.shape,dtype=np.float)
    sum_sq_diff2=np.zeros(p1.shape,dtype=np.float)
    numerator=np.zeros(p1.shape,dtype=np.float)
    denominator=np.zeros(p1.shape,dtype=np.float)
    diffs1=np.zeros(p1.shape,dtype=np.float)
    diffs2=np.zeros(p1.shape,dtype=np.float)
    p1_frame=np.pad(p1,support,mode='reflect')
    p2_frame=np.pad(p2,support,mode='reflect')
    
    means1[:,:]=ndimage.filters.uniform_filter(pic1[:,:,0],width)
    means2[:,:]=ndimage.filters.uniform_filter(pic2[:,:,0],width)
    for ind in range(numel):
        xoff=ind%width
        yoff=ind/width
        p1_off[:,:]=p1_frame[xoff:p1.shape[0]+xoff,yoff:p1.shape[1]+yoff]
        diffs1[:,:]=p1_off-means1
        sum_sq_diff1[:,:]=sum_sq_diff1+np.square(diffs1)
        p2_off[:,:]=p2_frame[xoff:p1.shape[0]+xoff,yoff:p1.shape[1]+yoff]
        diffs2[:,:]=p2_off-means2
        #print numerator
        sum_sq_diff2[:,:]=sum_sq_diff2+np.square(diffs2)
        numerator[:,:]=numerator+diffs1*diffs2
    sum_sq_diff1[:,:]=np.sqrt(sum_sq_diff1/numel)
    sum_sq_diff2[:,:]=np.sqrt(sum_sq_diff2/numel)
    numerator[:,:]=numerator/numel
    denominator[:,:]=sum_sq_diff1*sum_sq_diff2
    #denominator[denominator==0]=0.001
    return numerator/denominator
    

    
    
