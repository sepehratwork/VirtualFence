import numpy as np
import cv2
from time import time


def Intersection(frame, pts1, pts2, k=1):
	'''	
	Calculating intersection betweeen polygons with pts1 and pts2 in pts1
	'''
	# pts1 = np.array([pts1])
	# pts2 = np.array([pts2])

	# blank = np.zeros((700,700,3), dtype=np.uint8)

	# while True:
	#     cv2.imshow("blank",blank)
	#     if cv2.waitKey(0):
	#         break

	# copy each of the contours (assuming there's just two) to its own image. 
	# Just fill with a '1'.
	
	# s = time()
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img1 = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
	img1 = cv2.fillPoly(img1, [pts1], (127,127,127))
	# while True:
	#     cv2.imshow("img1",img1)
	#     if cv2.waitKey(0):
	#         break
	img1 = cv2.resize(img1, (0,0), fx=k, fy=k, interpolation=cv2.INTER_AREA)
	# img1 = img1.astype('int8')
	# while True:
	#     cv2.imshow("img1",img1)
	#     if cv2.waitKey(0):
	#         break

	img2 = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
	img2 = cv2.fillPoly(img2, [pts2], (127,127,127))
	# while True:
	#     cv2.imshow("img2",img2)
	#     if cv2.waitKey(0):
	#         break
	img2 = cv2.resize(img2, (0,0), fx=k, fy=k, interpolation=cv2.INTER_AREA)
	img2 = img2.astype('int8')
	# while True:
	#     cv2.imshow("img2",img2)
	#     if cv2.waitKey(0):
	#         break

	# blank = cv2.fillPoly(blank, [pts1], (127,127,127))
	# blank = cv2.fillPoly(blank, [pts2], (127,127,127))
	# while True:
	#     cv2.imshow("blank",blank)
	#     if cv2.waitKey(0):
	#         break

	total = 0
	total = int(total)
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			# for k in range(img1.shape[2]):
			if (img1[i][j] == 127):
				total += 1


	# now AND the two together
	# s = time()
	overlap = np.logical_and( img1, img2 )
	overlap = sum(sum(overlap))
	overlap = overlap / total
	overlap = overlap * 100
	# e1 = time()
	# print(f'overlap: {overlap} %, time: {e1 - s}')
	# overlap = int(overlap)
	# OR we could just add img1 to img2 and pick all points that sum to 2 (1+1=2):
	# s = time()

	# result = img1 + img2
	# while True:
	# 	cv2.imshow('Result', result)
	# 	if cv2.waitKey(0):
	# 		break
	
	# overlap2 = (img1+img2) == 254
	# overlap2 = sum(sum(overlap2))
	# overlap2 = overlap2 / total
	# overlap2 = overlap2 * 100
	# e2 = time()
	# print(f'overlap2: {overlap2}, time: {e2 - s}')

	# s = time()
	# overlap3 = 0
	# for i in range(img1.shape[0]):
	# 	for j in range(img1.shape[1]):
	# 		if (img1[i][j] == img2[i][j]) and (img1[i][j] == 127) and (img2[i][j] == 127):
	# 			overlap3 += 1
	# overlap3 = overlap3 / total
	# overlap3 = overlap3 * 100
	# e3 = time()
	# print(f'overlap3: {overlap3}, time: {e3 - s}')

	# overlap4 = np.zeros(blank.shape)
	# for i in range(blank.shape[0]):
	#     for j in range(blank.shape[1]):
	#         for k in range(blank.shape[2]):
	#             overlap4[i][j][k] = (img1[i][j][k] == img2[i][j][k])
	# overlap4 = sum(sum(sum(overlap4)))
	# overlap4 = overlap4 / total
	# overlap4 = overlap4 * 100

	# s = time()
	# result = img1 + img2
	# overlap5 = 0
	# for i in range(result.shape[0]):
	# 	for j in range(result.shape[1]):
	# 			if (result[i][j] == 254):
	# 				overlap5 += 1
	# overlap5 = overlap5 / total
	# overlap5 = overlap5 * 100
	# e5 = time()
	# print(f'overlap5: {overlap5}, time: {e5 - s}')
    
	return overlap


# # testing
# import DrawPolygon
# img = cv2.imread("Michael-Jordan-HD-Wallpapers-Download.jpg")

# img, pts1 = DrawPolygon.DrawPolygon(img)
# img, pts2 = DrawPolygon.DrawPolygon(img)

# overlap = Intersection(img, pts1, pts2, k=0.01)


# # Displaying the image
# # while(1):
# #     cv2.imshow('image', img)
# #     if cv2.waitKey(20) & 0xFF == 27:
# #         break
        
# cv2.destroyAllWindows()
