from ultralytics import YOLO
import cv2
import math
import numpy as np
from PIL import ImageGrab
import argparse

msg = "Detect objects on screen/webcam with yolov8 model"
parser = argparse.ArgumentParser(description = msg)
parser.add_argument("-m", "--model", help = "Path of model to train. Uses yolov8s if not passed.")
parser.add_argument("-c", "--classes", help = "Path of classes file. Uses annotations/classes.txt by default.")
parser.add_argument("-s", "--source", help = "Source of images to detect (screen, webcam).")
args = parser.parse_args()

if args.source == "webcam":
    #start webcam
    cap = cv2.VideoCapture(0)
    

# model
    if args.model:
        modelName = args.model
        print("Training model: % s" % args.model)
    else:
        print("Using default: yolov8s.pt")
        modelName = 'yolov8s.pt'

model = YOLO(modelName)
file_path = args.classes if args.classes else "annotations/classes.txt"
classNames = []

with open(file_path, "r") as file:
    for line in file:
        classNames.append(line.strip())

print("Classes: " + str(classNames))

while True:
    if args.source == "screen":
        im = ImageGrab.grab()
        a = np.array(im)
        screen = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        results = model(screen, stream=True)
    elif args.source == "webcam":
        success, screen = cap.read()
        results = model(screen, stream=True)
    else:
        raise Exception("Invalid image source")
    
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(screen, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1-10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(screen, classNames[cls] + " " + str(confidence), org, font, fontScale, color, thickness)

    cv2.imshow('my_screen', cv2.resize(screen, (960, 540)))
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()