import cv2

def extract(vid_name,frame_nums,results_dir):
    cap = cv2.VideoCapture(vid_name)
    for num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES,num-1)
        ret,frame=cap.read()
        cv2.imshow(str(num),frame)
        out_name=results_dir+vid_name+"-frame-num-"+str(num)+'.jpg'
        cv2.imwrite(out_name,frame)
