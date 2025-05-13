import io
import streamlit as st
import torch
from PIL import Image
import cv2
import os
import tempfile
import numpy as np
import base64
from ultralytics import YOLO
import threading

# ------------------- Background Setup -------------------
def get_base64_image(path):
    with open(path, "rb") as img_file:
        data = img_file.read()
    return base64.b64encode(data).decode()

image_path = "object_detection/photo-ai.jpeg"
base64_img = get_base64_image(image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- Sidebar Logo -------------------
@st.cache_resource
def display_logo(logo_path, title):
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as logo_file:
            encoded_logo = base64.b64encode(logo_file.read()).decode()

        st.sidebar.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{encoded_logo}" style="border-radius: 50%; width: 60px; height: 60px; margin-right: 10px;">
                <h3 style="margin: 0;">{title}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------------- Display Circular Logo and Title ----------------
def display_logo(logo_path, title):
    if os.path.exists(logo_path):
        encoded_logo = get_base64_image(logo_path)
        st.sidebar.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded_logo}" 
                     style="border-radius: 50%; width: 100px; height: 100px; object-fit: cover; margin-bottom: 10px;" />
                <h2 style="color: #6C63FF; font-size: 24px; margin: 0;">{title}</h2>
            </div>
        """, unsafe_allow_html=True)

logo_path = "object_detection\cinevo ai.jpg"  # Use forward slashes
display_logo(logo_path, "CINEVO-AI")

# ---------------- Divider ----------------
st.sidebar.markdown("<hr style='margin: 20px 0; border: 1px solid #ccc;'>", unsafe_allow_html=True)

# ---------------- Contact Links with Icons ----------------
st.sidebar.markdown("### 💼 Connect With Us")
st.sidebar.markdown("""
<style>
.link-list {
    list-style-type: none;
    padding-left: 10px;
    font-size: 16px;
}
.link-list li {
    margin-bottom: 10px;
}
.link-icon {
    width: 24px;
    height: 24px;
    vertical-align: middle;
    margin-right: 8px;
}
</style>

<ul class="link-list">
    <li>
        <img class="link-icon" src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
        <a href="https://github.com/yuvraj-3" target="_blank">GitHub: Yuvraj</a>
    </li>
    <li>
        <img class="link-icon" src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
        <a href="https://github.com/Shivam5571" target="_blank">GitHub: Shivam</a>
    </li>
    <li>
        <img class="link-icon" src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
        <a href="www.linkedin.com/in/shivam-verma-17683130a/" target="_blank">LinkedIn: Shivam</a>
    </li>
    <li>
        <img class="link-icon" src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
        <a href="https://www.linkedin.com/in/yuvraj-kumar-78a669323/" target="_blank">LinkedIn: Yuvraj</a>
    </li>
    <li>
        <img class="link-icon" src="https://cdn-icons-png.flaticon.com/512/561/561127.png">
        <a href="mailto:yuvrajkumarsingh303@gmail.com">Email: yuvrajkumarsingh303@gmail.com</a>
    </li>
</ul>
""", unsafe_allow_html=True)

# ---------------- Footer Divider ----------------
st.sidebar.markdown("<hr style='margin: 20px 0; border: 1px solid #ccc;'>", unsafe_allow_html=True)
# ------------------- Load YOLOv8 Model -------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolov8n.pt")
    model.to(device)
    return model

model = load_model()
st.title("🎬 CINEVO-AI")

# ------------------- Image Detection -------------------
option = st.selectbox("What would you like to detect?", ["Objects in Image", "Objects in Video", "Live Detection (Webcam)"])

# ------------------- Image Detection -------------------
if option == "Objects in Image":
    st.markdown("### 📷 Upload an Image")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="image_upload")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Detecting objects..."):
            results = model(image)
            annotated_image = results[0].plot()

        st.image(annotated_image, caption="Detected Objects", use_container_width=True)

        # Convert numpy.ndarray to PIL.Image and allow download
        pil_image = Image.fromarray(annotated_image)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        st.download_button("📥 Download Result", data=img_bytes, file_name="detected.jpg", mime="image/jpeg")


# ------------------- Video Detection -------------------
elif option == "Objects in Video":
    uploaded_video = st.file_uploader("🎥 Upload a Video", type=['mp4', 'mov', 'avi'], key="video_uploader")
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("🚀 Detect Objects in Video"):
            st.info("Processing video... please wait ⏳")

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = os.path.join('outputs', 'detected_output.mp4')
            os.makedirs('outputs', exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated = results[0].plot()
                out.write(annotated)
                frame_count += 1

            cap.release()
            out.release()

            st.success(f"✅ Done! Processed {frame_count} frames.")
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f,
                    file_name="detected_output.mp4",
                    mime="video/mp4"
                )


# ------------------- Live Webcam Detection -------------------
if option == "Live Detection (Webcam)":
    st.warning("🛑 This works only in a local environment with a connected webcam or phone camera.")
    
    cam_choice = st.radio("Select Camera Source:", ["Laptop Webcam", "External Webcam (USB)", "Phone Camera (IP Stream)"])

    # Default webcam index values
    if cam_choice == "Laptop Webcam":
        cam_index = 0
    elif cam_choice == "External Webcam (USB)":
        cam_index = 1  # Change this if needed (you can test other indexes like 2, 3...)
    else:
        cam_index = st.text_input("📱 Enter your IP camera stream URL (e.g., http://192.168.0.101:8080/video)")

    start_detection = st.button("🎥 Start Detection")
    stop_detection = st.button("🛑 Stop Detection")

    if start_detection:
        try:
            cap = cv2.VideoCapture(cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            stframe = st.empty()
            frame_skip = 4
            frame_count = 0
            st.info("🔴 Detection started. Click 'Stop Detection' to end.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Unable to read frame.")
                    break
                frame = cv2.flip(frame, 1)

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Perform object detection
                results = model(frame)
                annotated_frame = results[0].plot()

                # Use Streamlit to display images instead of OpenCV's imshow
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

                if stop_detection:
                    st.success("🛑 Detection stopped by user.")
                    break

            cap.release()

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
