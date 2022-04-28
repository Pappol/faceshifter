import time
import cv2
from imutils import face_utils
import dlib

cap = cv2.VideoCapture(0)
output_file = "/home/pi/Desktop/tirocinio/Video.mp4"

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

ret, frame = cap.read()
height, width, layers = frame.shape
fps = 30

p = "../preprocess/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# allow the camera to warmup
time.sleep(0.1)



# capture frames from the camera
while True:
    ret, frame = cap.read()
    #image = cv2.flip(image, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break
