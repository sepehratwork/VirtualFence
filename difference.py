import cv2
import imutils

# difference between two images

# reading images
path1 = '20210809_081648.jpg'
path2 = '20210809_081651.jpg'

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

# resizing images
img1 = cv2.resize(img1, (780, 540), interpolation = cv2.INTER_NEAREST)
img2 = cv2.resize(img2, (780, 540), interpolation = cv2.INTER_NEAREST)

# making a copy from first image and storing the difference in that
difference = cv2.imread(path1)
difference = cv2.resize(difference, (780, 540), interpolation = cv2.INTER_NEAREST)
cv2.absdiff(img1, img2, difference)

# converting the difference into gray image
difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

# increasing the size of the diffrence so that we can capture them all
for i in range(0, 3):
    dilated = cv2.dilate(difference_gray.copy(), None, iterations=i+1)

# threshold the gray image to binarise it. Anything pixel that has
# value more than 3 we are converting to white
# the image is called binarised as any value less than 3 will be 0 and
# all values equal to and more than 3 will be 255
(T, thresh) = cv2.threshold(dilated, 3, 255, cv2.THRESH_BINARY)

# finding countours
contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    difference = cv2.rectangle(difference, (x, y), (x+w, y+h), (0,255,0), 2)

while True:
    cv2.imshow("Difference",difference)
    if cv2.waitKey(0):
        break
