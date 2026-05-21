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
    "Detect cabbages from UAV imagery using YOLOv8. "
    "Select an image from the gallery or upload your own."
)

# =========================
# Model path
# =========================
MODEL_PATH = Path("models/best.pt")

@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Please place your trained YOLOv8 model file at: models/best.pt")
    st.stop()

model = load_model(str(MODEL_PATH))

# =========================
# Sidebar settings
# =========================
st.sidebar.header("⚙️ Detection settings")

conf_thres = st.sidebar.slider(
    "Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05
)
imgsz = st.sidebar.selectbox(
    "Inference image size", options=[640, 800, 1024, 1280], index=0
)
box_thickness = st.sidebar.slider(
    "Box line thickness", min_value=1, max_value=5, value=2, step=1
)
show_label = st.sidebar.checkbox("Show label & confidence", value=False)

# =========================
# Gallery image folder
# =========================
GALLERY_DIR = Path("hex_images_jpg")

def get_gallery_images() -> list[Path]:
    """Return sorted list of image files in the gallery folder."""
    if not GALLERY_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(
        [p for p in GALLERY_DIR.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.stem
    )

gallery_images = get_gallery_images()

# =========================
# Session state for selected image
# =========================
if "selected_gallery_path" not in st.session_state:
    st.session_state["selected_gallery_path"] = None

# =========================
# Gallery section
# =========================
if gallery_images:
    st.markdown("### 📂 画像ギャラリー（クリックで選択）")
    st.caption(f"{GALLERY_DIR} フォルダ内の画像: {len(gallery_images)} 枚")

    # Number of columns for the thumbnail grid
    THUMB_COLS = 7
    THUMB_SIZE = (160, 160)  # px for display

    rows = [gallery_images[i:i + THUMB_COLS]
            for i in range(0, len(gallery_images), THUMB_COLS)]

    for row in rows:
        cols = st.columns(len(row))
        for col, img_path in zip(cols, row):
            with col:
                # Load thumbnail
                try:
                    thumb = Image.open(img_path).convert("RGB")
                    thumb.thumbnail(THUMB_SIZE, Image.LANCZOS)
                except Exception:
                    st.warning(img_path.name)
                    continue

                # Highlight selected image with a green border
                is_selected = (
                    st.session_state["selected_gallery_path"] == str(img_path)
                )
                border_color = "#2ecc40" if is_selected else "#444"
                border_width = 3 if is_selected else 1

                # Wrap thumbnail in a styled container
                st.markdown(
                    f"""
                    <div style="
                        border: {border_width}px solid {border_color};
                        border-radius: 6px;
                        padding: 2px;
                        margin-bottom: 2px;
                        text-align: center;
                    ">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(thumb, use_container_width=True)

                # Button underneath each thumbnail
                btn_label = f"{'✅ ' if is_selected else ''}{img_path.stem}"
                if st.button(btn_label, key=f"btn_{img_path.stem}", use_container_width=True):
                    st.session_state["selected_gallery_path"] = str(img_path)
                    st.rerun()

    st.markdown("---")
else:
    st.info(
        f"ギャラリーフォルダ `{GALLERY_DIR}` が見つかりません。"
        "フォルダを作成して画像を入れると、ここにサムネイルが表示されます。"
    )

# =========================
# Image source selection
# =========================
st.markdown("### 🖼️ 検出する画像を選択")

# Determine default tab based on session state
if st.session_state["selected_gallery_path"]:
    default_tab = 0  # Gallery tab
else:
    default_tab = 1  # Upload tab

tab_gallery, tab_upload = st.tabs(["📂 ギャラリーから選択", "📤 画像をアップロード"])

selected_image: Image.Image | None = None
selected_name: str = ""

with tab_gallery:
    if st.session_state["selected_gallery_path"]:
        p = Path(st.session_state["selected_gallery_path"])
        st.success(f"選択中: **{p.name}**")
        try:
            selected_image = Image.open(p).convert("RGB")
            selected_name = p.stem
            # Preview of selected image
            preview_col, _ = st.columns([1, 2])
            with preview_col:
                st.image(selected_image, caption=p.name, use_container_width=True)
        except Exception as e:
            st.error(f"画像を読み込めません: {e}")
            selected_image = None
    else:
        st.info("上のギャラリーから画像をクリックして選択してください。")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an image for detection",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file).convert("RGB")
            selected_name = Path(uploaded_file.name).stem
            # If user uploads a file, clear gallery selection
            st.session_state["selected_gallery_path"] = None
        except Exception as e:
            st.error(f"Could not read the uploaded image: {e}")

# =========================
# Guard: no image selected
# =========================
if selected_image is None:
    st.info("ギャラリーから画像を選択するか、画像をアップロードしてください。")
    st.stop()

# =========================
# Run detection button
# =========================
st.markdown("---")
run_detection = st.button("▶️ 検出を実行", type="primary", use_container_width=True)

if not run_detection:
    st.stop()

# =========================
# Inference
# =========================
image_np = np.array(selected_image)
height, width = image_np.shape[:2]
st.write(f"画像サイズ: {width} × {height} px")

with st.spinner("Detecting with YOLO..."):
    results = model.predict(
        source=image_np,
        imgsz=imgsz,
        conf=conf_thres,
        verbose=False,
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
        zip(boxes_xyxy, confs, cls_ids), start=1
    ):
        x1, y1, x2, y2 = box
        box_width  = x2 - x1
        box_height = y2 - y1
        box_area   = box_width * box_height
        center_x   = (x1 + x2) / 2
        center_y   = (y1 + y2) / 2
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

        x1_i = max(0, min(int(x1), width - 1))
        y1_i = max(0, min(int(y1), height - 1))
        x2_i = max(0, min(int(x2), width - 1))
        y2_i = max(0, min(int(y2), height - 1))

        cv2.rectangle(
            annotated,
            (x1_i, y1_i),
            (x2_i, y2_i),
            color=(255, 0, 0),
            thickness=box_thickness,
        )

        if show_label:
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                annotated, label,
                (x1_i, max(y1_i - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 1, cv2.LINE_AA,
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
        file_name=f"cabbage_detection_{selected_name}.csv",
        mime="text/csv",
    )
else:
    st.warning("No cabbages detected. Try lowering the confidence threshold.")
