
# 🧠 CINEVO-AI: YOLOv8 Object Detection Web App

![Banner](object_detection/photo-ai.jpeg)

**CINEVO-AI** is a powerful, user-friendly web application built with **Streamlit** and **YOLOv8** by Ultralytics. It allows users to detect objects in **images**, **videos**, and **live webcam feeds**—including support for **external webcams** and **IP-based phone cameras**. It uses GPU acceleration (if available) and includes a beautiful UI with custom background, sidebar logo, contact links, and download options.

---

## 🚀 Features

- 🖼️ Image-based object detection  
- 🎥 Video-based object detection with frame-by-frame processing  
- 🔴 Real-time webcam detection (laptop, USB webcam, or IP stream)  
- 💻 GPU support via PyTorch CUDA (if available)  
- 📥 Download results (images/videos) after detection  
- 🌄 Background image with custom CSS  
- 🎨 Circular sidebar logo with social links  
- 📱 Mobile IP camera stream detection support  
- 🧠 Powered by [YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 📦 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```txt
streamlit
torch
ultralytics
Pillow
opencv-python
numpy
```

---

## 🛠️ Project Structure

```
object_detection/
│
├── app copy.py                  # Main Streamlit app
├── yolov8n.pt              # YOLOv8 model (auto-downloaded if not present)
├── photo-ai.jpeg           # Background image
├── cinevo ai.jpg           # Sidebar circular logo
└── outputs/                # Folder to store output videos
```

---

## ▶️ Run the App

```bash
streamlit run app copy.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📷 Usage Guide

### 1. **Detect in Image**
- Upload an image (`jpg/png`)
- Click **Detect**
- Download the result

### 2. **Detect in Video**
- Upload a video file (`mp4/avi/mov`)
- Click **Detect Objects in Video**
- Wait for processing and download the result

### 3. **Live Webcam Detection**
- Choose from:
  - Laptop Webcam (default)
  - External Webcam (change index if needed)
  - Phone Camera (via IP stream like `http://192.168.0.101:8080/video`)
- Start and stop live detection

> 💡 Tip: Use apps like **DroidCam** or **IP Webcam** on Android to stream phone camera

---

## 💻 Screenshots

### 📷 Image Detection  
![Image Detection](assets/screenshots/image_detection.jpg)

### 🎥 Video Detection  
![Video Detection](assets/screenshots/video_detection.jpg)

### 🔴 Live Detection  
![Live Webcam](assets/screenshots/webcam_detection.jpg)

---

## 🤝 Credits

- 🔍 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 🎨 Sidebar & UI by:  
  - [Shivam Verma](https://www.linkedin.com/in/shivam-verma-17683130a/)  
  - [Yuvraj Kumar](https://www.linkedin.com/in/yuvraj-kumar-78a669323/)

> GitHub:  
> - [Shivam](https://github.com/Shivam5571)  
> - [Yuvraj](https://github.com/yuvraj-3)

📧 Contact:  
- yuvrajkumarsingh303@gmail.com

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
