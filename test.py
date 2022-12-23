# import cv2
# import numpy as np

# # Python program to transform an image using
# # threshold.

# # Image operation using thresholding
# img = cv2.imread('morphology.jpg')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 0, 255,
# 							cv2.THRESH_BINARY_INV +
# 							cv2.THRESH_OTSU)

# thresh = cv2.resize(thresh, (780, 540), interpolation = cv2.INTER_NEAREST)


# # Noise removal using Morphological
# # closing operation
# kernel = np.ones((3, 3), np.uint8)
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
# 							kernel, iterations = 2)

# # Background area using Dialation
# bg = cv2.dilate(closing, kernel, iterations = 1)

# # Finding foreground area
# dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
# ret, fg = cv2.threshold(dist_transform, 0.02
# 						* dist_transform.max(), 255, 0)

# fg = cv2.resize(fg, (780, 540), interpolation = cv2.INTER_NEAREST)
# while True:
#     cv2.imshow('fg', fg)
#     cv2.imshow('thresh', thresh)
#     if cv2.waitKey(0):
#         break

# Python programe to illustrate
# Closing morphological operation
# on an image

# # organizing imports
# import cv2
# import numpy as np

# # return video from the first webcam on your computer.
# cap = cv2.VideoCapture(0)

# # loop runs if capturing has been initialized.
# while(1):
# 	# reads frames from a camera
# 	_, image1 = cap.read()
# 	_, image2 = cap.read()
	
# 	# Converts to HSV color space, OCV reads colors as BGR
# 	# frame is converted to hsv
# 	hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
# 	hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
	
# 	# defining the range of masking
# 	blue1 = np.array([110, 50, 50])
# 	blue2 = np.array([130, 255, 255])
	
# 	# initializing the mask to be
# 	# convoluted over input image
# 	mask = cv2.inRange(hsv1, blue1, blue2)

# 	# passing the bitwise_and over
# 	# each pixel convoluted
# 	res = cv2.bitwise_and(image1, image2)
# 	cv2.imshow('And', res)
	
# 	# defining the kernel i.e. Structuring element
# 	kernel = np.ones((5, 5), np.uint8)
	
# 	# defining the closing function
# 	# over the image and structuring element
# 	closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	
# 	# The mask and closing operation
# 	# is shown in the window
# 	cv2.imshow('Mask', mask)
# 	cv2.imshow('Closing', closing)
	
# 	# Wait for 'a' key to stop the program
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# # De-allocate any associated memory usage
# cv2.destroyAllWindows()

# # Close the window / Release webcam
# cap.release()


import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture('india.mp4')

ret, first_frame = cap.read()
background = first_frame.copy()

while ret:
	ret, frame1 = cap.read()

	frame_prime = np.float32(frame1)
	background_prime = np.float32(background)
	cv2.accumulateWeighted(frame_prime,background_prime, 0.01)
	background_prime = cv2.convertScaleAbs(background_prime)
	background = background_prime
	difference = background.copy()
	while True:
		cv2.imshow("Background",background)
		if cv2.waitKey(0):
			break

	difference = cv2.absdiff(background, frame1)
	difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
	while True:
		cv2.imshow('Threshold', difference_gray)
		if cv2.waitKey(0):
			break

	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(difference_gray,kernel,iterations = 1)
	while True:
		dilation = cv2.putText(dilation, 'Dilation', (10,55), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('Threshold', dilation)
		if cv2.waitKey(0):
			break
	
	(T, thresh) = cv2.threshold(dilation, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	while True:
		cv2.imshow('Threshold', thresh)
		if cv2.waitKey(0):
			break
	
	kernel = np.ones((9,9),np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
	while True:
		opening = cv2.putText(opening, 'Opening', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('Threshold', opening)
		if cv2.waitKey(0):
			break

	kernel = np.ones((9,9),np.uint8)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
	while True:
		closing = cv2.putText(closing, 'Closing', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('Threshold', closing)
		if cv2.waitKey(0):
			break
	
	contours = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	HighArea = 1000000
	LowArea = 20
	for contour in contours:
		if (cv2.contourArea(contour) > LowArea) & (cv2.contourArea(contour) < HighArea):
			(x, y, w, h) = cv2.boundingRect(contour)
			frame1 = cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
	while True:
		cv2.imshow('original', frame1)
		if cv2.waitKey(0):
			break

	if cv2.waitKey(0) == ord('q'):
		break
