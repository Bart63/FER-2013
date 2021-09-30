import numpy as np
import sys
import cv2 as cv

imagePath = sys.argv[1]
cascPath = sys.argv[2]

faceCascade = cv.CascadeClassifier(cascPath)

img  = cv.imread(imagePath)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 5,
    flags = cv.CASCADE_SCALE_IMAGE,
)


for (x, y, w, h) in face:
#   cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 8)
#                     start   end
    crop_img = img[y:y+h, x:x+w]


# Cleanup
cv.imshow("Image",  crop_img)
cv.waitKey(0)
cv.destroyAllWindows()