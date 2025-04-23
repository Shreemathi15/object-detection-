import random
import cv2
import numpy as np
from ultralytics import YOLO

# Load class names from coco.txt file
my_file = open("utils/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
my_file.close()

# Generate random colors for each class
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load the YOLOv8 model
model = YOLO("weights/yolov8n.pt", "v8")

# Load an image
frame = cv2.imread("inference/images/traffiv.png")

# Predict objects in the image
results = model.predict(source=[frame], conf=0.45, save=False)

# Process the detection results
if results and len(results[0].boxes) > 0:
    boxes = results[0].boxes
    for i in range(len(boxes)):
        box = boxes[i]  # Get each detected box
        clsID = int(box.cls.numpy()[0])  # Class ID
        conf = box.conf.numpy()[0]  # Confidence score
        bb = box.xyxy.numpy()[0]  # Bounding box coordinates

        # Draw the bounding box
        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[clsID],
            3,
        )

        # Display the class name and confidence
        label = f"{class_list[clsID]}: {round(conf, 2)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            label,
            (int(bb[0]), int(bb[1]) - 10),
            font,
            0.6,
            (255, 255, 255),
            2,
        )

# Show the image with detections
cv2.imshow("Image Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
