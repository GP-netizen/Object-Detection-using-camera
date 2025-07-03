import cv2
import numpy as np
import time
import os

# Debug: show working dir & files
print("CWD:", os.getcwd())
print("Files:", os.listdir(os.getcwd()))

# Load YOLO‑Tiny
net = cv2.dnn.readNet("yolov3-tiny.cfg", "yolov3-tiny.weights")
# (Optional: enable GPU if built with CUDA)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Get output layer names
try:
    output_layers = net.getUnconnectedOutLayersNames()
except AttributeError:
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

font = cv2.FONT_HERSHEY_SIMPLEX
start = time.time()
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    height, width = frame.shape[:2]

    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf > 0.3:
                # scale box coords back to image size
                cx, cy, w, h = (det[0]*width, det[1]*height,
                                det[2]*width, det[3]*height)
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(conf)
                class_ids.append(class_id)

    # Apply Non‑Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    # Draw detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            conf = confidences[i]
            color = [int(c) for c in colors[class_ids[i]]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x, y - 10), font, 0.6, color, 2)
            print(f"Detected: {label} ({conf:.2f})")

    # Calculate and show FPS
    fps = frame_id / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                font, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO-Tiny Detection", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
