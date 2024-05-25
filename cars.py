from ultralytics import YOLO
import cv2
import numpy as np
import cvzone

model = YOLO("../Yolo/yolov8l.pt")

cap = cv2.VideoCapture(r"D:\Videos\Cars Moving On Road Stock Footage - Free Download.mp4")

classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
           8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
           14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
           22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
           29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
           35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
           41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
           49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
           57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
           64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
           71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
           78: 'hair drier', 79: 'toothbrush'}

ids = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            corresponding_class = box.cls[0]
            if corresponding_class == 2:
                car_id = int(box.id.cpu().numpy()[0])
                ids.add(car_id)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = x2 - x1
                h = y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h), 5)
                cvzone.putTextRect(frame, f'{car_id}', (max(0, x1), max(35, y1)), 3)
    cvzone.putTextRect(frame, f'Car Count: {len(ids)}', pos=(0, 40))
    cv2.imshow('Window', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
