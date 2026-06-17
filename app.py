"""
Demo Klasifikasi Penyakit Daun Padi
Tab 1: MobileNetV2 (Transfer Learning)
Tab 2: CNN + SVM (Stacking/Hibrida)
"""

import os
import numpy as np
import cv2
import joblib
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model, Model

# =====================================================
#  KONFIGURASI
# =====================================================
IMAGE_SIZE = (224, 224)
CLASSES    = ['Sehat', 'Blight', 'Blast']

MOBILENET_PATH      = os.path.join("models", "best_MobileNetV2.h5")
CNN_EXTRACTOR_PATH  = os.path.join("models", "best_CNN_extractor.h5")
CNN_SVM_PKL_PATH    = os.path.join("models", "cnn_svm_classifier.pkl")

st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Padi",
    page_icon="🌾",
    layout="centered"
)

# =====================================================
#  HELPER: PREPROCESSING
# Harus identik dengan preprocessing saat training di notebook.
# =====================================================

def preprocess_image_raw(pil_image, image_size=IMAGE_SIZE):
    """
    Mereplikasi langkah preprocessing load_dataset() di notebook:
    resize -> median blur -> histogram equalization (YCrCb).
    Input : PIL Image (RGB)
    Output: array uint8 (H, W, 3), RGB, [0,255]
    """
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)

    # Noise reduction
    img = cv2.medianBlur(img, 3)

    # Histogram equalization per channel Y (YCrCb)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
    img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)

    return img  # uint8, [0,255]


def normalize_standard(img_uint8):
    """Untuk CNN+SVM: normalisasi /255.0 (sesuai notebook)."""
    return img_uint8.astype(np.float32) / 255.0


def normalize_imagenet(img_uint8):
    """Untuk MobileNetV2: normalisasi mean/std ImageNet (sesuai notebook)."""
    imgs = img_uint8.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (imgs - mean) / std


# =====================================================
#  HELPER: LOAD MODEL (cached supaya tidak reload tiap interaksi)
# =====================================================

@st.cache_resource
def load_mobilenet_model():
    if not os.path.exists(MOBILENET_PATH):
        return None
    return load_model(MOBILENET_PATH)


@st.cache_resource
def load_cnn_svm_bundle():
    if not (os.path.exists(CNN_EXTRACTOR_PATH) and os.path.exists(CNN_SVM_PKL_PATH)):
        return None, None, None

    model_cnn = load_model(CNN_EXTRACTOR_PATH)
    feature_extractor = Model(
        inputs=model_cnn.input,
        outputs=model_cnn.get_layer('feature_layer').output,
        name='Feature_Extractor'
    )

    bundle = joblib.load(CNN_SVM_PKL_PATH)
    scaler = bundle['scaler']
    svm    = bundle['svm']
    classes_pkl = bundle.get('classes', CLASSES)

    return feature_extractor, (scaler, svm), classes_pkl


# =====================================================
#  HELPER: PREDIKSI
# =====================================================

def predict_mobilenet(model, pil_image):
    img_raw = preprocess_image_raw(pil_image)
    img_norm = normalize_imagenet(img_raw)[np.newaxis, ...]
    probs = model.predict(img_norm, verbose=0)[0]
    return probs


def predict_cnn_svm(feature_extractor, scaler, svm, pil_image):
    img_raw = preprocess_image_raw(pil_image)
    img_norm = normalize_standard(img_raw)[np.newaxis, ...]

    feat = feature_extractor.predict(img_norm, verbose=0)
    feat_sc = scaler.transform(feat)

    pred_idx = svm.predict(feat_sc)[0]

    # SVC default tidak punya predict_proba kecuali probability=True saat training.
    # Coba ambil decision_function sebagai gambaran skor confidence relatif.
    try:
        probs = svm.predict_proba(feat_sc)[0]
        has_proba = True
    except AttributeError:
        scores = svm.decision_function(feat_sc)[0]
        # Normalisasi skor jadi seperti "confidence" relatif (bukan probabilitas asli)
        scores = scores - scores.min()
        probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(CLASSES)) / len(CLASSES)
        has_proba = False

    return pred_idx, probs, has_proba


def display_prediction(probs, classes, pred_label=None):
    pred_idx = int(np.argmax(probs)) if pred_label is None else classes.index(pred_label)
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    st.success(f"**Prediksi: {pred_class}**  ({confidence:.2f}%)")

    st.write("Detail skor per kelas:")
    for cls, p in zip(classes, probs):
        st.write(f"- {cls}: {p*100:.2f}%")
        st.progress(min(max(float(p), 0.0), 1.0))


# =====================================================
#  UI UTAMA
# =====================================================

st.title("🌾 Klasifikasi Penyakit Daun Padi")
st.caption("Sehat · Bacterial Leaf Blight (BLB) · Blast")

tab1, tab2 = st.tabs(["📱 MobileNetV2", "🔬 CNN + SVM"])

# ----------------- TAB 1: MobileNetV2 -----------------
with tab1:
    st.subheader("Klasifikasi dengan MobileNetV2 (Transfer Learning)")

    model_mn = load_mobilenet_model()

    if model_mn is None:
        st.error(
            f"File model tidak ditemukan di `{MOBILENET_PATH}`. "
            "Pastikan file .h5 sudah ditaruh di folder `models/`."
        )
    else:
        uploaded_mn = st.file_uploader(
            "Upload gambar daun padi",
            type=["jpg", "jpeg", "png"],
            key="uploader_mobilenet"
        )

        if uploaded_mn is not None:
            image = Image.open(uploaded_mn)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar input", use_container_width=True)
            with col2:
                with st.spinner("Memprediksi..."):
                    probs = predict_mobilenet(model_mn, image)
                display_prediction(probs, CLASSES)

# ----------------- TAB 2: CNN + SVM -----------------
with tab2:
    st.subheader("Klasifikasi dengan CNN + SVM (Stacking)")

    feature_extractor, svm_bundle, classes_pkl = load_cnn_svm_bundle()

    if feature_extractor is None:
        st.error(
            f"File model tidak ditemukan. Pastikan file berikut ada di folder `models/`:\n\n"
            f"- `{os.path.basename(CNN_EXTRACTOR_PATH)}`\n"
            f"- `{os.path.basename(CNN_SVM_PKL_PATH)}`"
        )
    else:
        scaler, svm = svm_bundle

        uploaded_cs = st.file_uploader(
            "Upload gambar daun padi",
            type=["jpg", "jpeg", "png"],
            key="uploader_cnnsvm"
        )

        if uploaded_cs is not None:
            image = Image.open(uploaded_cs)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar input", use_container_width=True)
            with col2:
                with st.spinner("Memprediksi..."):
                    pred_idx, probs, has_proba = predict_cnn_svm(
                        feature_extractor, scaler, svm, image
                    )
                if not has_proba:
                    st.caption(
                        "⚠️ Model SVM dilatih tanpa `probability=True`, "
                        "skor di bawah adalah confidence relatif (bukan probabilitas asli)."
                    )
                display_prediction(probs, classes_pkl)

st.markdown("---")
st.caption(
    "Catatan: hasil prediksi bersifat indikatif berdasarkan model machine learning, "
    "bukan diagnosis pasti. Untuk keputusan penanganan, konsultasikan dengan ahli "
    "pertanian/penyuluh setempat."
)
