# Object-Detection-using-camera
# YOLO‑Tiny Object Detection with OpenCV

Real‑time object detection from your webcam using YOLOv3‑Tiny and OpenCV’s DNN module.

---

## 📁 Repository Structure

Object_Detection/
├── main.py
├── yolov3-tiny.cfg
├── yolov3-tiny.weights
├── coco.names
└── README.md

markdown
Copy
Edit

- **main.py**  
  Python script that loads the YOLOv3‑Tiny model, captures frames from your webcam, runs inference, and draws bounding boxes & labels in real time.
- **yolov3-tiny.cfg**  
  Network configuration for the Tiny‑YOLOv3 model (416×416 input size).
- **yolov3-tiny.weights**  
  Pre‑trained Tiny‑YOLOv3 weights (COCO dataset, ~35 MB).
- **coco.names**  
  Class labels file (80 COCO object categories).
- **README.md**  
  Project documentation and usage instructions (this file).

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your‑username>/Object_Detection.git
cd Object_Detection
2. Install dependencies
Make sure you have Python 3.x installed. Then install required packages:

bash
Copy
Edit
pip install opencv-python numpy
Optional GPU acceleration
If you have a CUDA‑enabled OpenCV build, you can uncomment these lines in main.py:

python
Copy
Edit
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
3. Download (or verify) model files
This repo already includes:

yolov3-tiny.cfg

yolov3-tiny.weights

coco.names

If you need to re‑download them:

bash
Copy
Edit
curl -L -o yolov3-tiny.cfg \
  https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg

curl -L -o yolov3-tiny.weights \
  https://pjreddie.com/media/files/yolov3-tiny.weights

curl -L -o coco.names \
  https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
▶️ Usage
Run the detection script:

bash
Copy
Edit
python main.py
A window titled YOLO‑Tiny Detection will open showing your webcam feed.

Press q or Esc to quit.

You should see bounding boxes and labels for detected objects (e.g. “person”, “bottle”, “chair”) along with the real‑time FPS counter.

⚙️ Configuration
Confidence threshold
Adjust conf_threshold in the code (default 0.3) to filter out low‑confidence detections.

NMS threshold
Adjust nms_threshold (default 0.4) to control overlap suppression.

Input size
Change (416, 416) in blobFromImage() if you want a different resolution (trading off speed vs. accuracy).

📸 Screenshots
Add one or two screenshots of the detection in action (drag & drop into /docs folder and reference here).

📄 License
This project is released under the MIT License.

🙏 Acknowledgments
PJ Reddie’s Darknet for the YOLOv3‑Tiny model.

OpenCV DNN module for easy integration with YOLO.

Happy detecting!

markdown
Copy
Edit

**Next steps:**  
- Replace `<your-username>` with your GitHub username in the clone URL.  
- Optionally add an MIT `LICENSE` file.  
- Add screenshots under a `/docs` folder and reference them in the “Screenshots” section.  
- Push everything to GitHub, and your project page will look polished and complete!
