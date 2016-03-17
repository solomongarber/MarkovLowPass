import numpy as np

def plot(vec,ht):
    low=np.min(vec)
    vic=np.array(vec,dtype=np.int64)
    vic=vic-low
    high=np.max(vic)
    vic=(vic*ht)/high
    ans=np.zeros((ht,len(vec),3),dtype=np.uint8)
    for x in range(len(vec)):
        ans[ht-vic[x]-1:ht-vic[x]+1,x:x+2,:]=255
    return ans

def plot_sqz(vec,ht,wdth):
    low=np.min(vec)
    vic=np.array(vec,dtype=np.int64)
    vic=vic-low
    high=np.max(vic)
    x_inds=np.int32([x/(len(vec)/(wdth*1.)) for x in range(len(vec))])
    vic=(vic*ht)/high
    ans=np.zeros((ht+4,np.max(x_inds),3),dtype=np.uint8)
    for x in range(len(vec)):
        ytop=np.min((ht-vic[x]+1,ht-1))
        xtop=np.min((x_inds[x]+2,x_inds[x_inds.shape[0]-1]-1))
        ans[ht-vic[x]-1:ytop,x_inds[x]:xtop,:]=255
    return ans


def plot_peaks(vec,ht,wdth,peaks):
    low=np.min(vec)
    vic=np.array(vec,dtype=np.int64)
    vic=vic-low
    high=np.max(vic)
    denom=len(vec)/(wdth*1.)
    x_inds=np.int32([x/denom for x in range(len(vec))])
    vic=(vic*ht)/high
    ans=np.zeros((ht+4,np.max(x_inds),3),dtype=np.uint8)
    for x in range(len(vec)):
        ytop=np.min((ht-vic[x]+1,ht-1))
        xtop=np.min((x_inds[x]+2,x_inds[x_inds.shape[0]-1]-1))
        ans[ht-vic[x]-1:ytop,x_inds[x]:xtop,:]=255
    for peak in peaks:
        peak=int(peak/denom)
        ans[:,np.min((peak,ans.shape[1]-1)),1]=255
    return ans
    
    low=np.min(vec)
    vic=np.array(vec,dtype=np.int64)
    vic=vic-low
    high=np.max(vic)
    vic=(vic*ht)/high
    ans=np.zeros((ht,len(vec),3),dtype=np.uint8)
    up=True
    old=vic[0]
    for x in range(len(vec)):
        y=vic[x]
        if ((y<old) & up):
            ans[:,x-1,0]=255
        if y<old:
            up=False
        elif y>old:
            up=True
        ans[ht-vic[x]-1,x,:]=255
        old=y
    return ans

