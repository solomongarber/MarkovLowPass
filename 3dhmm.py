import math
import numpy as np
import cv2
#import dataQueue2
import denergyQueue2
import medQueue
import pyrStripe
stack_em=False
distance = 20
divisor=16
#divisor=20
smooth_exp = 2
local_energy = True
data_exp = 1
#distance_decay=.9
#smooth_lasso_exp=2
#smooth_local_energy = True
#smooth_local_exp = -1
#smooth_local_mult=200
data_local_mult=.5
data_local_exp=2
data_mult=1
skip_frames=100
smooth_mult=5
bad_fraction=3
median_support=1

results_dir='./med_pyr_maxmin/'

in_name='../Calc2FirstOrderDiffEqSepofVars.mp4'
start_name='calc_2'

#in_name='../subsamp-change-threshold-0.06-MH12non-euclid.avi'
#start_name='subsamp-change-threshold-0.06-MH12non-euclid'
#skip_frames=1

#in_name='../Quantizing-gravity.mp4'
#start_name='quantizing-grav'

#in_name='../Black-holes-by-Leonard-Susskind.mp4'
#start_name='black-holes'

#in_name='../DiffGeom18FrenetSerretEq.mp4'
#start_name='frenet-seret'

#in_name='../preProcessGPU.avi'
#start_name='pre-process'

#in_name='../non-euclid-subsamp-100.avi'
out_name=results_dir+'debugfour'+start_name+'-pyr-maxmin-block-delete-'+str(bad_fraction)+'-med-support'+str(median_support)+'-skip-'+str(skip_frames)+'-loc-mul-'+str(data_local_mult)+'-loc-exp'+str(data_local_exp)+'-twolevs-divisor-' +str(divisor)+'-1dhmm-dist-'+str(distance)+'-data_mult-'+str(data_mult)+'-data-exp'+str(data_exp)+'smooth-mult'+str(smooth_mult)+'-smooth_exp-'+str(smooth_exp)+"-stack_em"+str(stack_em)+'.mp4'


cap = cv2.VideoCapture(in_name)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
big_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fwidth=big_width/divisor
bad_diam=int(big_width/bad_fraction)
bad_sigma=bad_diam/3
bad_gaussian=cv2.getGaussianKernel(bad_diam+(1-np.mod(bad_diam,2)),bad_sigma)
bad_gaussian=np.reshape(bad_gaussian,len(bad_gaussian))
big_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fheight=big_height/divisor
fps=cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(out_name ,fourcc, fps, (big_width,big_height),True)


print (fwidth,fheight)
tot_time=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#tot_time=10000
ret = True

#num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_frames=tot_time
print num_frames
nD = 2*distance+1
num_pixels=fwidth*fheight
big_numpx=big_width*big_height
num_channels=3
if local_energy:
    curr_data=denergyQueue2.denergyQueue2((big_width,big_height,num_channels),(big_numpx,num_channels),(num_pixels,num_channels),divisor,num_pixels,big_numpx,nD,smooth_exp,smooth_mult,data_exp,data_mult,data_local_exp,data_local_mult,stack_em)
else:
    curr_data=dataQueue2.dataQueue2((big_width,big_height,num_channels),(big_numpx,num_channels),(num_pixels,num_channels),divisor,num_pixels,big_numpx,nD)
old_energy=np.zeros((num_pixels,nD),dtype=np.int)
new_energy=np.zeros((num_pixels,nD),dtype=np.int)
next_energy=np.zeros(old_energy.shape,dtype=np.int)
labels=np.zeros((num_pixels,nD),dtype=np.uint8)
denergy=np.zeros(num_pixels,dtype=np.int)
smenergy=np.zeros((num_pixels,nD),dtype=np.int)
time=0

bp=np.zeros((num_pixels,nD,num_frames),dtype=np.uint8)

def get_dispute(inframe,outframe):
    diff_vec=np.sum(np.sum(np.power(np.abs(np.array(inframe,dtype=np.int)-np.array(outframe,dtype=np.int)),2),2),0)
    #diff_vec=np.reshape(diff_vec,diff_vec.shape[0])
    diff_window=np.convolve(diff_vec,bad_gaussian,mode='same')
    m=np.argmax(diff_window)
    bad_radius=bad_diam/2
    s_width=bad_diam/2
    ll_cap=np.max((0,m-bad_radius-s_width))
    lr_cap=np.max((0,m-bad_radius))
    rl_cap=m+bad_radius
    range1=diff_vec[ll_cap:lr_cap]
    range2=diff_vec[rl_cap:np.min((big_width,m+bad_radius+s_width))]
    a=0
    b=big_width
    if(range1.shape[0]>0):
        a=ll_cap+np.argmin(range1)
    if(range2.shape[0]>0):
        b=rl_cap+np.argmin(range2)
    #if(m-bad_radius-s_width<0):
    #    a=0
    #if(m+bad_radius+s_width
    return (a,b)
    #return (m-bad_radius,m+bad_radius)
    
while(cap.isOpened() and ret):
    print time
    cap.set(cv2.CAP_PROP_POS_FRAMES,skip_frames*time);
    ret, frame = cap.read()
    if ret:
        curr_data.add_frame(frame)
        for label in range(nD):
            label_frame=curr_data.get_frame(label+1)
            denergy[:]=curr_data.get_dcost(label)
            smenergy[:,:]=curr_data.get_scost(label_frame)
            new_energy[:,:]=old_energy+smenergy
            labels[:,label]=np.array(np.argmin(new_energy,1),dtype=np.uint8)
            next_energy[:,label]=new_energy[range(num_pixels),labels[:,label]]+denergy
    bp[:,:,time]=labels
    old_energy[:,:]=next_energy[:,:]
    time=time+1
    if time*skip_frames >= tot_time:
        break


time=time-nD-2
final_label=np.array(np.argmin(old_energy,1),dtype=np.uint8)
prev_labels=np.zeros(num_pixels,dtype=np.uint8)
prev_labels[:]=labels[range(num_pixels),final_label]
#outframe=cv2.cvtColor(curr_data.get_output(final_label),3)
outframe=np.zeros(frame.shape,dtype=np.uint8)
outframe[:,:,:]=curr_data.get_output(final_label)
inframe=np.zeros(frame.shape,dtype=np.uint8)
print curr_data.now.shape
inframe[:,:,:]=np.reshape(curr_data.now,frame.shape)
print "good"
medframe=np.zeros(frame.shape,dtype=np.uint8)
medframe[:,:,:]=outframe
vid_cache=pyrStripe.pyrStripe(outframe)
writeframe=np.zeros(frame.shape,dtype=np.uint8)
writeframe[:,:,:]=outframe
stripeframe=np.zeros(frame.shape,dtype=np.uint8)
stripeframe[:,:,:]=outframe
med=medQueue.medQueue(median_support,frame.shape,big_numpx,num_channels)
#print outframe
out.write(outframe)
for t in range(time,-1,-1):
    print 'a'
    labels[:,:]=bp[:,:,t+nD]
    print 'b'
    curr_data.add_frame_left()
    print 'c'
    outframe[:,:,:]=curr_data.get_output(prev_labels)
    print 'd'
    inframe[:,:,:]=np.reshape(curr_data.now,frame.shape)
    print 'e'
    (a,b)=get_dispute(inframe,outframe)
    print 'f'
    #medframe[:,:,:]=vid_cache.stripeOut(inframe,a,b)
    stripeframe[:,:,:]=vid_cache.stripeOut(inframe,a,b)
    print 'g'
    medframe[:,:,:]=inframe[:,:,:]
    print 'h'
    medframe[:,a:b,:]=0
    print 'i'
    if median_support>1:
        med.add_frame(medframe)
        writeframe[:,:,:]=med.get_median()
        out.write(writeframe)
    else:
        writeframe[:,:,:]=medframe
    print 'j'
    out.write(inframe)
    print 'k'
    out.write(outframe)
    print 'l'
    out.write(writeframe)
    print 'm'
    out.write(stripeframe)
    print 'n'
    prev_labels[:]=labels[range(num_pixels),prev_labels]
    print t
    
    

#f.close()
cap.release()
cv2.destroyAllWindows()
