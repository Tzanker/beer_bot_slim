from image_processing import get_coords
import cv2

hMin=117
sMin = 151
vMin=0
hMax=180
sMax = 255
vMax=180


# Load image

image = cv2.imread('image.jpg')
image = cv2.resize(image, (960, 540))

print(get_coords(hMin, sMin, vMin, hMax, sMax, vMax, image))