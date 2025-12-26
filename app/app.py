import os
#os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

@st.cache_resource
def load_mask_model():
    model_path = os.path.join(os.getcwd(), "outputs", "models", "mask_detection_model.h5")
    return tf.keras.models.load_model(model_path)

class VideoProcessor:
    def __init__(self):
        self.model = load_mask_model()
        self.class_names = ["Incorrect Mask", "With Mask", "Without Mask"]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        
        preds = self.model.predict(img_array, verbose=0)
        idx = np.argmax(preds[0])
        label = self.class_names[idx]
        conf = preds[0][idx] * 100
        
        color = (0, 255, 0) if idx == 1 else (0, 165, 255) if idx == 0 else (0, 0, 255)
        cv2.putText(img, f"{label} {conf:.1f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 10)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="Mask Detection Pro", layout="wide")
    
    st.markdown("""
        <style>
        .stApp { background-color: #000000; color: #FFFFFF; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: #111; color: white; border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True)

    st.title("Face Mask Detector")
    
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = 0

    tab1, tab2 = st.tabs(["📤 UPLOAD", "🎥 REAL-TIME"])

    with tab1:
        st.subheader("Upload Image")
        file = st.file_uploader("Upload...", type=["jpg","png","jpeg"], key="uploader")
        
        if file:
            img = Image.open(file).convert("RGB")
            col_in, col_out = st.columns(2)
            
            with col_in:
                st.image(img, caption="Original Image", use_column_width=True)
            
            if st.button("RUN ANALYSIS", key="analyze_static"):
                with st.spinner("Analyzing image..."):
                    model = load_mask_model()
                    class_names = ["Incorrect Mask", "With Mask", "Without Mask"]
                    
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    preds = model.predict(img_array, verbose=0)
                    idx = np.argmax(preds[0])
                    label = class_names[idx]
                    conf = preds[0][idx] * 100
                    
                    color_hex = "#00FF00" if idx == 1 else "#FFA500" if idx == 0 else "#FF0000"
                    
                    with col_out:
                        st.markdown(f"""
                            <div style="background-color: #111111; padding: 20px; border-radius: 10px; border-left: 10px solid {color_hex};">
                                <h4 style="margin:0; color: #888888;">RESULT</h4>
                                <h2 style="color: {color_hex}; margin: 0;">{label.upper()}</h2>
                                <p style="margin:0;">Confidence: {conf:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        draw = img.copy()
                        ImageDraw.Draw(draw).rectangle([0, 0, draw.width, draw.height], outline=color_hex, width=25)
                        st.image(draw, caption="Detection Result", use_column_width=True)

    with tab2:
        st.subheader("Live Monitoring")
        webrtc_streamer(
            key="mask-detection-v3",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )

if __name__ == "__main__":

    main()
