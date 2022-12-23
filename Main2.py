import cv2
import numpy as np
import imutils
import DrawPolygon
import Intersection
import Margin
from time import time


def Surveillance(file_path=0,
                dilation_kernel=3, dilation_iteration=1,
                threshold=45, 
                opening_kernel=3, opening_iteration=1, 
                closing_kernel=3, closing_iteration=1, 
                HighArea=300, LowArea=85, 
                Alpha=0.01, Background=None, 
                IntersectionCalc=False, Tracking=False):
    '''
    -   Getting the path of the video and by assessing the Virtual Fence it starts to surveilling the picture and detecting the defference between background.
    -   First you should assess the virtual fence.
        Then you should click 'Esc' for going into next frames and starting surveilling.
    -   'file_path' argument is your video path. If you do not give anything it will start getting frames from your local webcam!
    -   'dilation' argument makes your differeces bigger as it grows up
    -   'threshold' argument is how sensitive you want to be for finding defferences between the present frame and background picture.
    -   'k' argument is the size of your kernel for erosion
    -   'iteration' is the number of repetition of erosion on your difference image
    -   'Area' arguments limit the range of the area of your contours. If a contour is out of these amounta it will not be considered.
    -   'Alpha' argument determines how fast your background picture updates.
    -   'IntersectionCalc' argument alows to do the calculation for finding which contour has intersection with your virtual fence, if it becomes 'True'. It will slow the proccess a lot! :( 
    -   'Tracking' argument alows the code to do the tracking for the next frame.
    '''
    statics = [[],[],[]]
    # Assessing the Virtual Fence for the first frame
    # Using the 'pts' for next frames
    cap = cv2.VideoCapture(file_path)
    grabbed, first_frame = cap.read()

    # using background for the detection comparision
    background = first_frame.copy()
    # first_frame = cv2.resize(first_frame, (780, 540), interpolation = cv2.INTER_NEAREST)
    first_frame, pts = DrawPolygon.DrawPolygon(first_frame)

    if pts != []:
        margin = Margin.Margin(pts, 100)
        margin = np.int32(margin)
    else:
        pts = []
        margin = []

    while grabbed:
        
        # in first frame we detect
        grabbed, frame1 = cap.read()
        if grabbed == False:
            break
        # frame1 = cv2.resize(frame2, (780, 540), interpolation = cv2.INTER_NEAREST)
        
        # in second frame we track
        grabbed, frame2 = cap.read()
        if grabbed == False:
            break
        # frame2 = cv2.resize(frame1, (780, 540), interpolation = cv2.INTER_NEAREST)
        if Background == None:
            # updating background after each frame
            frame_prime = np.float32(frame1)
            background_prime = np.float32(background)
            cv2.accumulateWeighted(frame_prime,background_prime,Alpha)
            background_prime = cv2.convertScaleAbs(background_prime)
            background = background_prime
        else:
            background = cv2.imread(Background)


        # making a copy from first image and storing the difference in that
        # difference = cv2.resize(difference, (780, 540), interpolation = cv2.INTER_NEAREST)
        s = time()
        difference = cv2.absdiff(frame1, background)

        # background = cv2.polylines(background, [pts],
                                # True, (255,0,0), 2)
        frame1 = cv2.polylines(frame1, [pts],
                                True, (0,0,255), 2)
        frame2 = cv2.polylines(frame2, [pts],
                                True, (0,0,255), 2)

        # converting the difference into gray image
        difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        # while True:
        # difference_gray = cv2.putText(difference_gray, 'Difference', (5,5), cv2.FONT_HERSHEY_SIMPLEX, 
        #                         1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Difference', difference_gray)
            # if cv2.waitKey(0):
                # break
        
        # Dilation
        dilation_kernel = int(abs(dilation_kernel))
        kernel = np.ones((dilation_kernel,dilation_kernel),np.uint8)
        dilated = cv2.dilate(difference_gray, kernel, iterations=dilation_iteration)
        # while True:
        # dilated = cv2.putText(dilated, 'Dilation', (5,45), cv2.FONT_HERSHEY_SIMPLEX, 
        #                         1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Dilation', dilated)
            # if cv2.waitKey(0):
                # break
        
        # Thresholding
        # threshold = int(abs((threshold/100) * 255))
        (T, thresh) = cv2.threshold(dilated, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # while True:
        # thresh = cv2.putText(thresh, 'Threshold', (5,65), cv2.FONT_HERSHEY_SIMPLEX, 
                                # 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Threshold', thresh)
            # if cv2.waitKey(0):
                # break
        
        # Opening
        opening_kernel = int(abs(opening_kernel))
        kernel = np.ones((opening_kernel,opening_kernel),np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_iteration)
        # while True:
        # opening = cv2.putText(opening, 'Opening', (5,25), cv2.FONT_HERSHEY_SIMPLEX, 
        #                         1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Opening', opening)
            # if cv2.waitKey(0):
                # break

        # Closing
        closing_kernel = int(abs(closing_kernel))
        kernel = np.ones((closing_kernel,closing_kernel),np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=closing_iteration)
        # while True:
        # closing = cv2.putText(closing, 'Closing', (5,85), cv2.FONT_HERSHEY_SIMPLEX, 
                                # 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Closing', closing)
            # if cv2.waitKey(0):
                # break

        # finding countours
        contours = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            if (cv2.contourArea(contour) > LowArea) & (cv2.contourArea(contour) < HighArea):
                (x, y, w, h) = cv2.boundingRect(contour)
                # difference = cv2.rectangle(difference, (x, y), (x+w, y+h), (0,255,0), 2)
                rectangle = np.array([(x,y), (x, y+h), (x+w, y+h), (x+w, y)])
                if IntersectionCalc == True:
                    s1 = time()
                    if Intersection.Intersection(frame1, rectangle, margin) > 5:
                        if Intersection.Intersection(frame1, rectangle, pts) > 20:
                            frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,0,255), 2)
                        else:
                            frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,255), 2)
                    else:
                        frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
                    t1 = time()
                    t1 = t1 - s1
                    print(f'Intersection: {t1}')
                    statics[2].append(t1)
                else:
                    frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)

                if Tracking == True:
                    # tracking
                    s2 = time()
                    tracker = cv2.TrackerKCF_create()
                    # Initialize tracker with first frame and bounding box
                    (x, y, w, h) = cv2.boundingRect(contour)
                    contour = (x, y, x+w, y+h)
                    grabbed = tracker.init(frame1, contour)
                    # Update contour by tracker
                    grabbed, new_contour = tracker.update(frame2)
                    if grabbed:
                        # Tracking success
                        x = int(new_contour[0])
                        y = int(new_contour[1])
                        w = int(new_contour[2])
                        h = int(new_contour[3])
                    t2 = time()
                    t2 = t2 - s2
                    print(f'Tracking: {t2}')
                    statics[1].append(t2)
                
                rectangle = np.array([(x,y), (x, y+h), (x+w, y+h), (x+w, y)])
                if IntersectionCalc == True:
                    s11 = time()
                    if Intersection.Intersection(frame2, margin, rectangle) > 10:
                        if Intersection.Intersection(frame2, pts, rectangle) > 10:
                            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,0,255), 2)
                        else:
                            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,255,255), 2)
                    else:
                        frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,255,0), 2)
                    t11 = time()
                    t11 = t11 - s11
                    print(f'Intersection: {t11}')
                    statics[2].append(t11)
                else:
                    frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,255,0), 2)

        t = time()
        t = t - s
        print(f'2 Frames: {t}')
        statics[0].append(t)

        # while True:
        cv2.imshow("Surveillance",frame1)
            # if cv2.waitKey(0):
                # break
        # while True:
        cv2.imshow("Surveillance",frame2)
            # if cv2.waitKey(0):
                # break
        
        # while True:
        cv2.imshow("Background",background)
            # if cv2.waitKey(0):
                # break
        
        k = cv2.waitKey(10)
        if k == ord('q'):
            return statics
    
    return statics


file_path = "14Min.mp4"
statics = Surveillance(file_path=file_path,
                        dilation_kernel=3, dilation_iteration=2,
                        threshold=20,
                        opening_kernel=5, opening_iteration=2,
                        closing_kernel=3, closing_iteration=2, 
                        HighArea=3000, LowArea=200, Alpha=0.001, Background="14MinBG.jpg", 
                        IntersectionCalc=True, Tracking=False)

# statics = Surveillance(dilation_kernel=3, dilation_iteration=1, threshold=45, opening_kernel=3, opening_iteration=1, closing_kernel=3, closing_iteration=1, HighArea=300, LowArea=85, Alpha=0.01, IntersectionCalc=False, Tracking=False)

print(f'Average proccessing per two pixels:                                                             {sum(statics[0])/len(statics[0])}')
if len(statics[1]) != 0:
    print(f'Average proccessing for tracking of each bounding box:                                          {sum(statics[1])/len(statics[1])}')
if len(statics[2]) != 0:
    print(f'Average proccessing for calculating the interaction of each bounding box with virtual fence:    {sum(statics[2])/len(statics[2])}')
