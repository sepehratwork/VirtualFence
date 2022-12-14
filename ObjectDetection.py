# import the necessary packages
import numpy as np
# import argparse
import time
import cv2
# import os

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-y", "--yolo", required=True,
# 	help="base path to YOLO directory")
# # ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
# 	help="threshold when applying non-maxima suppression")
# args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
# labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
# weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
weightsPath = "yolov4-tiny.weights"
# configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
configPath = "yolov4-tiny.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
# print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

video_path = "34Min.mp4"
cap = cv2.VideoCapture(video_path)
isOpen, frame = cap.read()
background_original = cv2.imread("34MinBG.jpg")

a = [[],[]]

while isOpen:
    start = time.time()
    
    # load our input image and grab its spatial dimensions
    # image = cv2.imread(args["image"])
    isOpen, frame = cap.read()
    (H, W) = frame.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    # start = time.time()
    layerOutputs = net.forward(ln)
    # end = time.time()

    # show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            # if confidence > args["confidence"]:
            if confidence > 0.4:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    #     args["threshold"])
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4,
        0.3)

    background = background_original.copy()

    objects = 0
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            objects += 1
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

            # print(f'Object {objects}')

            for i in range(x-5, x+w+5):
                for j in range(y-5, y+h+5):
                    background[i][j] = frame[i][j]

    # show the output image
    cv2.imshow("Image", frame)
    cv2.imshow("Background", background)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    a[0].append((end - start))
    a[1].append(objects)

print(sum(a[0])/len(a[0]))
print(sum(a[1])/len(a[1]))
