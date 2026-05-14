from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Cabbage Detection App",
    layout="wide"
)

st.title("🥬 Cabbage Detection App")
st.caption(
    "Detect cabbages from UAV imagery using YOLOv8 and display bounding boxes with a results table."
)


# =========================
# Model path
# =========================
# Streamlit Cloud用:
# app.py と同じ階層に models フォルダを作り、その中に best.pt を置く
#
# Folder structure:
# cabbage_streamlit_app/
# ├─ app.py
# ├─ requirements.txt
# └─ models/
#    └─ best.pt

MODEL_PATH = Path("models/best.pt")


@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model only once."""
    return YOLO(model_path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Please place your trained YOLOv8 model file at: models/best.pt"
    )
    st.stop()

model = load_model(str(MODEL_PATH))


# =========================
# Sidebar settings
# =========================
st.sidebar.header("⚙️ Detection settings")

conf_thres = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

imgsz = st.sidebar.selectbox(
    "Inference image size",
    options=[640, 800, 1024, 1280],
    index=0
)

box_thickness = st.sidebar.slider(
    "Box line thickness",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)

show_label = st.sidebar.checkbox(
    "Show label & confidence",
    value=False
)


# =========================
# Image upload
# =========================
uploaded_file = st.file_uploader(
    "Upload an image for detection",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is None:
    st.info("Upload an image to start detection.")
    st.stop()


# =========================
# Read image
# =========================
try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error("Could not read the uploaded image.")
    st.exception(e)
    st.stop()

image_np = np.array(image)
height, width = image_np.shape[:2]

st.write(f"Image size: {width} × {height} px")


# =========================
# Inference
# =========================
with st.spinner("Detecting with YOLO..."):
    results = model.predict(
        source=image_np,
        imgsz=imgsz,
        conf=conf_thres,
        verbose=False
    )

result = results[0]


# =========================
# Extract detections
# =========================
detections = []
annotated = image_np.copy()

if result.boxes is not None and len(result.boxes) > 0:
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    for i, (box, conf, cls_id) in enumerate(
        zip(boxes_xyxy, confs, cls_ids),
        start=1
    ):
        x1, y1, x2, y2 = box

        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        class_name = model.names.get(cls_id, str(cls_id))

        detections.append({
            "id": i,
            "class": class_name,
            "confidence": round(float(conf), 4),
            "x1_px": round(float(x1), 1),
            "y1_px": round(float(y1), 1),
            "x2_px": round(float(x2), 1),
            "y2_px": round(float(y2), 1),
            "center_x_px": round(float(center_x), 1),
            "center_y_px": round(float(center_y), 1),
            "box_width_px": round(float(box_width), 1),
            "box_height_px": round(float(box_height), 1),
            "box_area_px2": round(float(box_area), 1),
        })

        x1_i, y1_i, x2_i, y2_i = map(
            int,
            [x1, y1, x2, y2]
        )

        # Prevent coordinates from exceeding image boundaries
        x1_i = max(0, min(x1_i, width - 1))
        y1_i = max(0, min(y1_i, height - 1))
        x2_i = max(0, min(x2_i, width - 1))
        y2_i = max(0, min(y2_i, height - 1))

        # Draw bounding box
        # image_np is RGB, so (255, 0, 0) gives a red box
        cv2.rectangle(
            annotated,
            (x1_i, y1_i),
            (x2_i, y2_i),
            color=(255, 0, 0),
            thickness=box_thickness
        )

        if show_label:
            label = f"{class_name} {conf:.2f}"

            cv2.putText(
                annotated,
                label,
                (x1_i, max(y1_i - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )


df = pd.DataFrame(detections)


# =========================
# Display results
# =========================
st.subheader("Detection results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Original image")
    st.image(image_np, use_container_width=True)

with col2:
    st.markdown("#### Detection result")
    st.image(annotated, use_container_width=True)

st.markdown("---")

st.metric("Cabbages detected", len(df))

if len(df) > 0:
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Download detection CSV",
        data=csv,
        file_name="cabbage_detection_results.csv",
        mime="text/csv"
    )

else:
    st.warning(
        "No cabbages detected. Try lowering the confidence threshold."
    )