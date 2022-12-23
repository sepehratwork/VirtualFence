import torch
from PIL import Image
import cv2

model = torch.load('yolox_nano.pth')
# model.eval()

# img = Image.open('Michael-Jordan-HD-Wallpapers-Download.jpg')
# image = data_transform(img).unsqueeze(0)

img = cv2.imread('Michael-Jordan-HD-Wallpapers-Download.jpg')

output = model(img)

cv2.imshow(output)
cv2.waitKey(0)
