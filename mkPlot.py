import numpy as np

def plot(vec,ht):
    low=np.min(vec)
    vic=np.array(vec,dtype=np.int64)
    vic=vic-low
    high=np.max(vic)
    vic=(vic*ht)/high
    ans=np.zeros((ht,len(vec),3),dtype=np.uint8)
    for x in range(len(vec)):
        ans[ht-vic[x]-1,x,:]=255
    return ans
    
def plot_peaks(vec,ht):
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

