import os
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pathlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CLASS_NAMES = ["Incorrect Mask", "With Mask", "Without Mask"]

@st.cache_resource
def load_mask_model():
    BASE_DIR = pathlib.Path(__file__).parent.parent
    model_path = BASE_DIR / "outputs" / "models" / "mask_detection_model.h5"

    if not model_path.exists():
        st.error("Model not found. Please train the model first.")
        return None

    return tf.keras.models.load_model(model_path)

class VideoProcessor:
    def __init__(self):
        self.model = load_mask_model()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

        preds = self.model(img_array, training=False).numpy()
        idx = np.argmax(preds[0])
        label = CLASS_NAMES[idx]
        conf = preds[0][idx] * 100

        color = (0, 200, 0) if idx == 1 else (255, 165, 0) if idx == 0 else (255, 60, 60)

        cv2.putText(
            img,
            f"{label} ({conf:.1f}%)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 8)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(
        page_title="Face Mask Detection",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0f1117;
            color: #eaeaea;
        }
        .sidebar-title {
            font-size: 22px;
            font-weight: 600;
        }
        .subtle {
            color: #9aa0a6;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("**Face Mask Detection**")
        st.markdown(
            "<p class='subtle'>Deep Learning based image classification & object detection</p>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        mode = st.radio(
            "Choose Detection Mode",
            ["📤 Upload Image", "🎥 Real-time Camera"]
        )

        st.markdown("---")

    if mode == "📤 Upload Image":
        st.subheader("Upload Image")

        file = st.file_uploader(
            "Upload an image (JPG / PNG / JPEG)",
            type=["jpg", "jpeg", "png"]
        )

        if file:
            img = Image.open(file).convert("RGB")
            col1, col2 = st.columns(2)

            with col1:
                st.image(img, caption="Original Image", use_column_width=True)

            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    model = load_mask_model()

                    if model:
                        img_resized = img.resize((224, 224))
                        img_array = np.array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

                        preds = model.predict(img_array, verbose=0)
                        idx = np.argmax(preds[0])
                        label = CLASS_NAMES[idx]
                        conf = preds[0][idx] * 100

                        color_hex = "#2ecc71" if idx == 1 else "#f39c12" if idx == 0 else "#e74c3c"

                        with col2:
                            st.markdown(
                                f"""
                                <div style="
                                    background:#161b22;
                                    padding:24px;
                                    border-radius:14px;
                                    border-left:8px solid {color_hex};
                                ">
                                    <h4 style="color:#aaa;">RESULT</h4>
                                    <h2 style="color:{color_hex};">{label.upper()}</h2>
                                    <p>Confidence: <b>{conf:.2f}%</b></p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.progress(int(conf))

                            draw = img.copy()
                            ImageDraw.Draw(draw).rectangle(
                                [0, 0, draw.width, draw.height],
                                outline=color_hex,
                                width=18,
                            )
                            st.image(draw, caption="Detection Result", use_column_width=True)

    else:
        st.subheader("Real-time Camera Detection")
        st.markdown(
            "<p class='subtle'>Allow camera access and keep your face visible</p>",
            unsafe_allow_html=True
        )

        webrtc_streamer(
            key="mask-detection-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
        )

if __name__ == "__main__":
    main()

