```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
```


```python
background = None
acc_Wt = 0.5

#Defining region of intrest in the frame
roi_top = 20
roi_bot = 300
roi_right = 300
roi_left = 600
```


```python
def calc_acc_Wt(frame,acc_Wt):
    global background

    # Initially pass background as frame
    if background is None:
        background = frame.copy().astype(float)
    return None
    
    cv2.accumulateWeighted(background,frame,acc_Wt)
```


```python
def segment(frame,threshold = 25):
    global background

    diff = cv2.absdiff(background.astype(np.uint8),frame)

    _, thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours,key = cv2.contourArea)

    return(thresholded, hand_segment)
    
```


```python
def count_fingers(thresholded,hand_segment):

    conv_hull = cv2.convexHull(hand_segment)

    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

    cX = (left[0]+right[0]) // 2
    cY = (top[0]+bottom[0]) // 2

    distance = pairwise.euclidean_distances(X=[(cX,cY)],Y=[top,bottom,left,right])[0]
    max = distance.max()

    radius = int(0.8*max)
    circumference = (2*np.pi*radius)

    circular_roi = np.zeros(thresholded.shape[:2],dtype = np.uint8)

    cv2.circle(circular_roi,(cX,cY),radius,255,10)

    circular_roi = cv2.bitwise_and(thresholded,thresholded,mask = circular_roi)

    contours, hierarchy = cv2.findContours(circular_roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)

        out_of_wrist = ((cY+(cY+h)) < (y+h))

        limit_points = ((circumference*0.25)>cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1
    

    return count        
```


```python
cap = cv2.VideoCapture(0)
num_frame = 0
while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bot, roi_right:roi_left]

    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)

    if num_frame < 60:
        calc_acc_Wt(gray, acc_Wt)
        if num_frame <= 59:
            cv2.putText(frame_copy,"Calculating running average",(200,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.imshow("Finger Count",frame_copy)
    else:
        hand = segment(gray)
        if hand is not None:

            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),1)
            fingers = count_fingers(thresholded,hand_segment)

            cv2.putText(frame_copy,str(fingers),(70,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            cv2.imshow('Thresholded',thresholded)
    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bot),(0,0,255),5)

    num_frame += 1

    cv2.imshow("Finger Count",frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
```
