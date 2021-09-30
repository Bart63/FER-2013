import cv2
import sys

# cl arguments
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# creating cascade from given file
faceCascade = cv2.CascadeClassifier(cascPath)

# read image and convert it to grayscale
img = cv2.imread(imagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# applying haar cascade to grayscale image
faces = haarCascade.detectMultiScale(
    gray, 
    scaleFactor=1.1,
    minNeighbors=5,
    flags = cv2.CASCADE_SCALE_IMAGE,
    minSize=(48, 48)
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Mordy", img)
cv2.waitKey(0)