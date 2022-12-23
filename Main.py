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
                HighArea=40000, LowArea=9000, 
                Alpha=0.01, Background='1', 
                IntersectionCalc=True, s=0.5, Tracking=False):
    '''
    -   Getting the path of the video and by assessing the Virtual Fence it starts to surveilling the picture by detecting the defference between background.
    -   First you should assess the virtual fence.
    -   'file_path' argument is your video path. If you do not give anything it will start getting frames from your local webcam!
    -   'dilation_kernel' & 'opening_kernel' & 'closing_kernel' arguments are the kernel of your morphology operations.
    -   'dilation_iteration' & 'opening_iteration' & 'closing_iteration' arguments are the number of iteration(s) of your morphology operations .
    -   'threshold' argument is how sensitive you want to be for finding defferences between the present frame and background picture.
    -   'Area' arguments limit the range of the area of your contours. If a contour is out of these amounta it will not be considered.
    -   'Alpha' argument determines how fast your background picture updates.
    -   'Background' is the path of the 
    -   'IntersectionCalc' argument alows to do the calculation for finding which contour has intersection with your virtual fence, if it becomes 'True'. It will slow the proccess a lot! :( 
    -   's' argument is a scale to simplify the intersection calculation. It's a number between 0 to 1. The lower it becomes, the simplerer the calculation becomes.
    -   'Tracking' argument alows the code to do the tracking for the next frame.
    '''
    statics = [[],[],[]]
    # Assessing the Virtual Fence for the first frame
    # Using the 'pts' for next frames
    cap = cv2.VideoCapture(file_path)
    grabbed, first_frame = cap.read()
    _, second_frame = cap.read()

    # first_frame = cv2.resize(first_frame, (780, 540), interpolation = cv2.INTER_AREA)
    first_frame, pts = DrawPolygon.DrawPolygon(first_frame)

    if pts != []:
        margin = Margin.Margin(pts, 60)
        margin = np.int32(margin)
    else:
        pts = []
        margin = []

    # for yolo
    # load the COCO class labels our YOLO model was trained on
    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    
    # derive the paths to the YOLO weights and model configuration
    weightsPath = "yolov4-tiny.weights"
    configPath = "yolov4-tiny.cfg"

    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    
    while grabbed:
        start = time()
        # in first frame we detect
        grabbed, frame1 = cap.read()
        if grabbed == False:
            break
        (H1, W1) = frame1.shape[:2]
        # frame1 = cv2.resize(frame2, (780, 540), interpolation = cv2.INTER_AREA)
        
        # in second frame we track
        grabbed, frame2 = cap.read()
        if grabbed == False:
            break
        (H2, W2) = frame2.shape[:2]
        # frame2 = cv2.resize(frame1, (780, 540), interpolation = cv2.INTER_AREA)
        
        if Background == None:
            # updating background after each frame
            background = first_frame.copy()
            frame_prime = np.float32(frame1)
            background_prime = np.float32(background)
            cv2.accumulateWeighted(frame_prime,background_prime,Alpha)
            background_prime = cv2.convertScaleAbs(background_prime)
            background = background_prime
        
        elif Background == '1':
            background = second_frame.copy()
        
        else:
            background_original = cv2.imread(Background)
            background = background_original.copy()

        # here comes yolo
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # # associated probabilities
        # blob = cv2.dnn.blobFromImage(frame1, 1 / 255.0, (416, 416),
        #                                 swapRB=True, crop=False)
        # net.setInput(blob)
        # start2 = time()
        # layerOutputs = net.forward(ln)
        # end = time()

        # # show timing information on YOLO
        # print("[INFO] YOLO took {:.6f} seconds".format(end - start2))

        # # initialize our lists of detected bounding boxes, confidences, and
        # # class IDs, respectively
        # boxes = []
        # confidences = []
        # classIDs = []

        # # loop over each of the layer outputs
        # for output in layerOutputs:
        #     # loop over each of the detections
        #     for detection in output:
        #         # extract the class ID and confidence (i.e., probability) of
        #         # the current object detection
        #         scores = detection[5:]
        #         classID = np.argmax(scores)
        #         confidence = scores[classID]
                
        #         # filter out weak predictions by ensuring the detected
        #         # probability is greater than the minimum probability
        #         # if confidence > args["confidence"]:
        #         if confidence > 0.4:
        #             # scale the bounding box coordinates back relative to the
        #             # size of the image, keeping in mind that YOLO actually
        #             # returns the center (x, y)-coordinates of the bounding
        #             # box followed by the boxes' width and height
        #             box = detection[0:4] * np.array([W1, H1, W1, H1])
        #             (centerX, centerY, width, height) = box.astype("int")
                    
        #             # use the center (x, y)-coordinates to derive the top and
        #             # and left corner of the bounding box
        #             x = int(centerX - (width / 2))
        #             y = int(centerY - (height / 2))
                    
        #             # update our list of bounding box coordinates, confidences,
        #             # and class IDs
        #             boxes.append([x, y, int(width), int(height)])
        #             confidences.append(float(confidence))
        #             classIDs.append(classID)

        # # apply non-maxima suppression to suppress weak, overlapping bounding
        # # boxes
        # # idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        # idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        # # adding detected objects to background
        # if len(idxs) > 0:
        #     for i in idxs.flatten():
        #         (x, y) = (boxes[i][0], boxes[i][1])
        #         (w, h) = (boxes[i][2], boxes[i][3])
        #         # frame = frame1.copy()
        #         # background[y:y+h,x:x+w,:] = 0
        #         background[y-3:y+h+3,x-3:x+w+3,:] = frame1[y-3:y+h+3,x-3:x+w+3,:]
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(background, (x, y), (x + w, y + h), color, 2)
                # for i in range(frame1.shape[0]):
                #     for j in range(frame1.shape[1]):
                #         if (i >= (x-5)) & (j >= (y-5)) & (i <= (x+w+5)) & (j <= (y+h+5)):
                #             background[i][j] = frame1[i][j]
                #         else:
                #             pass
        
        # for the objects that yolo didn't detect
        # making a copy from first image and storing the difference in that
        # difference = cv2.resize(difference, (780, 540), interpolation = cv2.INTER_AREA)
        difference = cv2.absdiff(frame1, background)

        # background = cv2.polylines(background, [pts],
                                # True, (255,0,0), 2)
        frame1 = cv2.polylines(frame1, [pts],
                                True, (255,0,0), 2)
        frame2 = cv2.polylines(frame2, [pts],
                                True, (255,0,0), 2)

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

        # ensure at least one detection exists
        idxs = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                color = (0, 255, 0)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
                if Tracking == False:
                    # color = [int(c) for c in COLORS[classIDs[i]]]
                    color = (0, 255, 0)
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                
                if Intersection == True:
                    s1 = time()
                    rectangle = np.array([(x,y), (x, y+h), (x+w, y+h), (x+w, y)])
                    if Intersection.Intersection(frame1, rectangle, margin, k=s) > 50:
                        if Intersection.Intersection(frame1, rectangle, pts, k=s) > 50:
                            frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,0,255), 2)
                            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                            cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,0,255), 2)

                        else:
                            frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,255), 2)
                            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                            cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,255,255), 2)
                    else:
                        frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 2)
                    t1 = time()
                    t1 = t1 - s1
                    print(f'Intersection: {t1}')
                    statics[2].append(t1)
                else:
                    frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                
                if Tracking == True:
                    # tracking
                    s2 = time()
                    tracker = cv2.TrackerKCF_create()
                    # Initialize tracker with first frame and bounding box
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
                else:
                    frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,102,255), 2)
                
                rectangle = np.array([(x,y), (x, y+h), (x+w, y+h), (x+w, y)])
                if IntersectionCalc == True:
                    s11 = time()
                    if Intersection.Intersection(frame2, rectangle, margin, k=s) > 50:
                        if Intersection.Intersection(frame2, rectangle, pts, k=s) > 50:
                            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,0,255), 2)
                            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                            cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,0,255), 2)
                        else:
                            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,255,255), 2)
                            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                            cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,255,255), 2)
                    else:
                        frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 2)
                    t11 = time()
                    t11 = t11 - s11
                    print(f'Intersection: {t11}')
                    statics[2].append(t11)
                else:
                    frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

        for contour in contours:
            if (cv2.contourArea(contour) > LowArea) & (cv2.contourArea(contour) < HighArea):
                (x, y, w, h) = cv2.boundingRect(contour)
                difference = cv2.rectangle(difference, (x, y), (x+w, y+h), (0,255,0), 2)
                rectangle = np.array([(x,y), (x, y+h), (x+w, y+h), (x+w, y)])
                if IntersectionCalc == True:
                    s1 = time()
                    if Intersection.Intersection(frame1, rectangle, margin, k=s) > 50:
                        if Intersection.Intersection(frame1, rectangle, pts, k=s) > 50:
                            frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,0,255), 2)
                            text = "Unknown"
                            cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,0,255), 2)

                        else:
                            frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,255), 2)
                            text = "Unknown"
                            cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,255,255), 2)
                    else:
                        frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (255,255,255), 2)
                        text = "Unknown"
                        cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,255,255), 2)
                    t1 = time()
                    t1 = t1 - s1
                    print(f'Intersection: {t1}')
                    statics[2].append(t1)
                else:
                    frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (255,255,255), 2)
                    text = "Unknown"
                    cv2.putText(frame1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,255), 2)

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
                else:
                    frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (255,255,255), 2)
                    text = "Unknown"
                    cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,255), 2)
                
                rectangle = np.array([(x,y), (x, y+h), (x+w, y+h), (x+w, y)])
                if IntersectionCalc == True:
                    s11 = time()
                    if Intersection.Intersection(frame2, rectangle, margin, k=s) > 50:
                        if Intersection.Intersection(frame2, rectangle, pts, k=s) > 50:
                            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,0,255), 2)
                            text = "Unknown"
                            cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,0,255), 2)
                        else:
                            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,255,255), 2)
                            text = "Unknown"
                            cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,255,255), 2)
                    else:
                        frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (255,255,255), 2)
                        text = "Unknown"
                        cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,255,255), 2)
                    t11 = time()
                    t11 = t11 - s11
                    print(f'Intersection: {t11}')
                    statics[2].append(t11)
                else:
                    frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (255,255,255), 2)
                    text = "Unknown"
                    cv2.putText(frame2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,255), 2)
        
        end = time()
        t = end - start
        statics[0].append(t)
        print(f'2 Frames: {t}')

        # while True:
        cv2.imshow("Surveillance",frame1)
            # if cv2.waitKey(10):
                # break
        # while True:
        cv2.imshow("Surveillance",frame2)
            # if cv2.waitKey(10):
                # break
        
        # while True:
        cv2.imshow("Background",background)
            # if cv2.waitKey(10):
                # break
        
        q = cv2.waitKey(10)
        if q == ord('q'):
            return statics
    
    return statics


file_path = "14min.mp4"
statics = Surveillance(file_path=file_path,
                        dilation_kernel=5, dilation_iteration=2,
                        threshold=45,
                        opening_kernel=7, opening_iteration=1,
                        closing_kernel=5, closing_iteration=1,
                        HighArea=500000, LowArea=5000, Alpha=0.001, Background='1',
                        IntersectionCalc=True, s=0.1, Tracking=False)


print(f'Average proccessing per two frames:                                                             {sum(statics[0])/len(statics[0])}')
if len(statics[1]) != 0:
    print(f'Average proccessing for tracking of each bounding box:                                          {sum(statics[1])/len(statics[1])}')
if len(statics[2]) != 0:
    print(f'Average proccessing for calculating the interaction of each bounding box with virtual fence:    {sum(statics[2])/len(statics[2])}')
