import numpy as np
import cv2
from math import pi

def ExpectedValue(f):
    mean = np.mean(f)
    sigma = np.sqrt(np.var(f))
    distribution = (1/(sigma*np.sqrt(2*pi))) * (np.exp(-(((f-mean)/sigma)/2)))
    E = sum(distribution*f)
    return E, mean, sigma, distribution

# video = cv2.VideoCapture("14Min.mp4")

# FOI = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

# #creating an array of frames from frames chosen above
# frames = []
# for frameOI in FOI:
#     video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
#     ret, frame = video.read()
#     frames.append(frame)

# E, mean, sigma, distribution = ExpectedValue(frames)
# print(len(distribution))
# while True:
#     cv2.imshow("Gaussian", E)
#     if cv2.waitKey(0):
#         break
# while True:
#     cv2.imshow("Gaussian", mean)
#     if cv2.waitKey(0):
#         break
# while True:
#     cv2.imshow("Gaussian", sigma)
#     if cv2.waitKey(0):
#         break


# #calculate the average
# backgroundFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
# backgroundFrame = Exception(frames)
# # cv2.imwrite("bg.jpg",backgroundFrame)
# while True:
#     cv2.imshow("Background Frame",backgroundFrame)
#     if cv2.waitKey(0):
#         break

import numpy as np
import cv2
# from skimage import data, filters

# Open Video
cap = cv2.VideoCapture('india.mp4')

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)
# cv2.imwrite('34MinBG.jpg', medianFrame)