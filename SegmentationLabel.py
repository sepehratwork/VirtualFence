import numpy as np
import cv2
import os
from numpy.core.defchararray import count

from numpy.core.numeric import isclose
import DrawPolygon

path = ''
counter = 0
for img in os.listdir(path + '/*.jpg'):
    counter += 1
    image = cv2.imread(img)
    image, pts = DrawPolygon.DrawPolygon(image)
    mask = np.zeros(image.shape)
    
    isClosed = True
    color = (255, 255, 255)
    thickness = -1
    mask = cv2.polylines(mask, [pts],
                        isClosed, color, thickness)
    cv2.imwrite(f'Mask{counter}', mask)
