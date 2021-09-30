import cv2
import sys

# cl arguments
cascPath = sys.argv[1]

# cascade creation
faceCascade = cv2.CascadeClassifier(cascPath)

# setting source to default cam
vidCapture = cv2.VideoCapture(0)

# frame by frame capture
# return code(ret) doesn't matter when reading from webcam
while True:
    ret, frame = vidCapture.read()

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = faceCascade.detectMultiScale(
    gray, 
    scaleFactor=1.1,
    minNeighbors=5,
    flags = cv2.CASCADE_SCALE_IMAGE,
    minSize=(48, 48)
    )

    # splice detected face
    for (x, y, w, h) in faces:
        crop_frame = frame[y:y+h, x:x+w]
        print(faces)
    # show cropped frame
    cv2.imshow('Webcam capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vidCapture.release()
cv2.destroyAllWindows()