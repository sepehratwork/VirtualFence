import numpy as np
import cv2
from collections import Counter, defaultdict
import imutils

file_path ='20210808_143227.mp4'
cap = cv2.VideoCapture(file_path)
grabbed, frame = cap.read()
frame = imutils.resize(frame,width =500)
img = cv2.imwrite("shot.jpg",frame)
firstframe_path ='shot.jpg'

firstframe = cv2.imread(firstframe_path)

firstframe_blur = cv2.GaussianBlur(firstframe,(21,21),0)
firstframe_gray = cv2.cvtColor(firstframe_blur, cv2.COLOR_BGR2GRAY)


cv2.namedWindow('CannyEdgeDet',cv2.WINDOW_NORMAL)
cv2.namedWindow('Abandoned Object Detection',cv2.WINDOW_NORMAL)
cv2.namedWindow('Morph_CLOSE',cv2.WINDOW_NORMAL)


cap = cv2.VideoCapture(0)

consecutiveframe=5

track_temp=[]
track_master=[]
track_temp2=[]

top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)

frameno = 0
# roi = cv2.selectROI(frame,showCrosshair=False)
# cv2.destroyWindow('ROI selector')
# firstframe = firstframe[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
while (cap.isOpened()):
    ret, frame = cap.read()
    #frame = imutils.resize(frame,width =500)
    #roi = cv2.selectROI(frame,showCrosshair=False)
    #forig = frame.copy()
    #frame=frame[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]


    if ret==0:
        break

    frameno = frameno + 1
    #cv2.putText(frame,'%s%.f'%('Frameno:',frameno),(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2) 





    frame_diff = cv2.absdiff(firstframe, frame)
    while True:
        cv2.imshow('frame_diff',frame_diff)
        if cv2.waitkey(0):
            break
    frame_gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)


    edged = cv2.Canny(frame_blur,10,200)
    while True:
        cv2.imshow('CannyEdgeDet',edged)
        if cv2.waitkey(0):
            break
    kernel2 = np.ones((5,5),np.uint8)
    thresh2 = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, 
    kernel2,iterations=2)
    #cv2.imshow('Morph_Close', thresh2)


    cnts,_ = cv2.findContours(thresh2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 


    mycnts =[]
    for c in cnts:

        M = cv2.moments(c)
        if M['m00'] == 0: 
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])


            if cv2.contourArea(c) < 300: ##or cv2.contourArea(c)>20000:
                pass
            else:
                mycnts.append(c)

                (x, y, w, h) = cv2.boundingRect(c)
                sumcxcy=cx+cy
                #print(sumcxcy)



                #track_list.append(cx+cy)
                track_temp.append([cx+cy,frameno])


                track_master.append([cx+cy,frameno])
                #print("did:",did,frameno)
                countuniqueframe = set(j for i, j in track_master) # get a set of unique frameno. then len(countuniqueframe) 

                #print("pip:",countuniqueframe)

                #------------------------------------------------------- 
                # ---------
                # Store history of frames ; no. of frames stored set by 
                #'consecutiveframe' ;
                # if no. of no. of unique frames > consecutiveframes, 
                #then 'pop or remove' the earliest frame ; defined by
                # minframeno. Objective is to count the same values 
                #occurs in all the frames under this list. if yes, 
                 # it is likely that it is a stationary object and not a 
                 #passing object (walking) 
                 # And the value is stored separately in 
                 #top_contour_dict , and counted each time. This dict is 
                 #the master
                 # dict to store the list of suspecious object. Ideally, 
                 #it should be a short list. if there is a long list
                 # there will be many false detection. To keep the list 
                 #short, increase the 'consecutiveframe'.
                 # Keep the number of frames to , remove the 
                 #minframeno.; but hard to remove, rather form a new list 
                 #without
                 #the minframeno.
                 #------------------------------------------------------ 
                 #----------
                if len(countuniqueframe)>consecutiveframe or False: 
                    minframeno=min(j for i, j in track_master)
                    for i, j in track_master:
                        if j != minframeno: # get a new list. omit the  those with the minframeno

                            track_temp2.append([i,j])

                    track_master=list(track_temp2) # transfer to the 
                                                    # master list
                    track_temp2=[]


                #print 'After',track_master

                #count each of the sumcxcy
                #if the same sumcxcy occurs in all the frames, store in  master contour dictionary, add 1


                countcxcy = Counter(i for i, j in track_master)
                #print countcxcy
                #example countcxcy : Counter({544: 1, 537: 1, 530: 1, 
                #523: 1, 516: 1})
                #if j which is the count occurs in all the frame, store 
                #the sumcxcy in dictionary, add 1
                for i,j in countcxcy.items(): 
                    if j>=consecutiveframe:
                        top_contour_dict[i] += 1


                if sumcxcy in top_contour_dict:
                    if top_contour_dict[sumcxcy]>50:
                        cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 0), 3) 

                        cv2.putText(frame,'%s'%('foreign_Object'),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2) 


                        # cv2.putText(frame, "foreign object !!!", (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) 

                        print ('Detected : ', sumcxcy,frameno,obj_detected_dict) 


                        # Store those objects that are detected, and store 
                        #the last frame that it happened.
                        # Need to find a way to clean the top_contour_dict, 
                        #else contour will be detected after the 
                        # object is removed because the value is still in 
                        #the dict.
                        # Method is to record the last frame that the object 
                        #is detected with the Current Frame (frameno)
                        # if Current Frame - Last Frame detected > some big 
                        #number say 100 x 3, then it means that 
                        # object may have been removed because it has not 
                        #been detected for 100x3 frames.

                        obj_detected_dict[sumcxcy]=frameno

    for i, j in obj_detected_dict.items():
        if frameno - obj_detected_dict[i]>200:
            print ('PopBefore',i,obj_detected_dict[i],frameno,obj_detected_dict)

            print ('PopBefore : top_contour:',top_contour_dict)
            obj_detected_dict.pop(i) 


            # Set the count for eg 448 to zero. because it has not be 
            #'activated' for 200 frames. Likely, to have been removed.
            top_contour_dict[i]=0
            print('PopAfter',i,obj_detected_dict[i],frameno,obj_detected_dict)  

            print ('PopAfter : top_contour :',top_contour_dict)



    #cv2.rectangle(forig, (roi[0], roi[1]),(roi[0]+roi[2],roi[1]+roi[3]), (0, 255, 0),1) 

    while True:
        cv2.imshow('Detections',frame)
        if cv2.waitKey(0):
            break



        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()