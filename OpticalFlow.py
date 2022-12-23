import cv2
import numpy as np
from time import time

# # Sparse Optical Flow

# cap = cv2.VideoCapture('14Min.mp4')

# # params for corner detection
# feature_params = dict( maxCorners = 1000,
# 					qualityLevel = 0.05,
# 					minDistance = 2,
# 					blockSize = 2 )

# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize = (21, 21),
# 				maxLevel = 2,
# 				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
# 							10, 0.03))

# # Create some random colors
# color = np.random.randint(0, 255, (1000, 3))

# # Take first frame and find corners in it
# ret, old_frame = cap.read()

# old_gray = cv2.cvtColor(old_frame,
# 						cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
# 							**feature_params)

# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)

# while(1):
	
#     ret, frame = cap.read()
#     frame_gray = cv2.cvtColor(frame,
# 							cv2.COLOR_BGR2GRAY)

# 	# calculate optical flow
#     s = time()
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
# 										frame_gray,
# 										p0, None,
# 										**lk_params)

# 	# Select good points
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]

# 	# draw the tracks
#     for i, (new, old) in enumerate(zip(good_new,
# 									good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
# 						color[i].tolist(), 2)
		
#         frame = cv2.circle(frame, (int(a), int(b)), 5,
# 						color[i].tolist(), -1)

#     print(time() - s)	
#     img = cv2.add(frame, mask)

#     while True:
#         cv2.imshow('frame', img)
#         if cv2.waitKey(0):
#             break
	
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break

# 	# Updating Previous frame and points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)

# cv2.destroyAllWindows()
# cap.release()







# Dense Optical Flow

# The video feed is read in as
# a VideoCapture object
cap = cv2.VideoCapture("busy.mp4")

# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
ret, first_frame = cap.read()

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 255

while(cap.isOpened()):
	
	# ret = a boolean return value from getting
	# the frame, frame = the current frame being
	# projected in the video
    ret, frame = cap.read()
	
	# Opens a new window and displays the input
	# frame
    while(True):
        cv2.imshow("input", frame)
        if cv2.waitKey(0):
            break
	
	# Converts each frame to grayscale - we previously
	# only converted the first frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Calculates dense optical flow by Farneback method
    s = time()
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
									None,
									0.5, 3, 15, 3, 5, 1.2, 0)
	
	# Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	
	# Sets image hue according to the optical flow
	# direction
    mask[..., 0] = angle * 180 / np.pi / 2
	
	# Sets image value according to the optical flow
	# magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
	
	# Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    print(f'Dense Optical Flow: {time() - s}')

	# Opens a new window and displays the output frame
    while True:
        cv2.imshow("dense optical flow", rgb)
        if cv2.waitKey(0):
            break
	
	# Updates previous frame
    prev_gray = gray
	
	# Frames are read by intervals of 1 millisecond. The
	# programs breaks out of the while loop when the
	# user presses the 'q' key
    if(cv2.waitKey(0) & 0xFF == ord('q')):
        break

# The following frees up resources and
# closes all windows
cap.release()
cv2.destroyAllWindows()



