import cv2
import numpy as np

c = cv2.VideoCapture(0)
_,f = c.read()

avg1 = np.float32(f)
avg2 = np.float32(f)

while(1):
    _,f = c.read()
	
    cv2.accumulateWeighted(f,avg1,0.1)
    cv2.accumulateWeighted(f,avg2,0.01)
	
    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    while True:
        cv2.imshow('img',f)
        if cv2.waitKey(0):
            break
    
    while True:
        cv2.imshow('avg1',res1)
        if cv2.waitKey(0):
            break
    
    while True:
        cv2.imshow('avg2',res2)
        if cv2.waitKey(0):
            break
    
    if cv2.waitKey(0) == ord('q'):
        break
   

cv2.destroyAllWindows()
c.release()