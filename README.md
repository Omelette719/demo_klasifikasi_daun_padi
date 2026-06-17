# Demo Klasifikasi Penyakit Daun Padi

Demo Streamlit dengan 2 tab:
- **Tab 1**: MobileNetV2 (Transfer Learning)
- **Tab 2**: CNN + SVM (Stacking)

## 1. Persiapan

### a. Install Python
Pastikan Python 3.10 atau 3.11 sudah terinstall di laptop kamu (cek dengan `python --version` di terminal/CMD).
TensorFlow 2.18 belum mendukung Python 3.13 ke atas — kalau versi Python kamu lebih baru, install Python 3.11 dulu atau gunakan virtual environment dengan versi tersebut.

### b. (Opsional tapi disarankan) Buat virtual environment
```bash
python -m venv venv

# Aktifkan:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### c. Install dependency
```bash
pip install -r requirements.txt
```

## 2. Siapkan file model

Salin 3 file model kamu ke dalam folder `models/`, dengan nama PERSIS seperti ini:

```
models/
├── best_MobileNetV2.h5
├── best_CNN_extractor.h5
└── cnn_svm_classifier.pkl
```

Kalau nama file aslimu berbeda, ada dua pilihan:
1. Rename filenya supaya sesuai nama di atas, ATAU
2. Edit path di `app.py` bagian `MOBILENET_PATH`, `CNN_EXTRACTOR_PATH`, `CNN_SVM_PKL_PATH` di awal file, sesuaikan dengan nama file kamu.

## 3. Jalankan demo

```bash
streamlit run app.py
```

Browser akan otomatis terbuka ke `http://localhost:8501`. Kalau tidak otomatis terbuka, buka manual URL tersebut.

## 4. Cara pakai

1. Pilih tab (MobileNetV2 atau CNN+SVM).
2. Upload gambar daun padi (.jpg/.jpeg/.png).
3. Hasil prediksi dan skor per kelas akan langsung muncul.

## Troubleshooting

**Error "No module named tensorflow" / dependency lain**
Pastikan virtual environment sudah diaktifkan sebelum `pip install` dan sebelum `streamlit run`.

**Error saat load .h5 ("unknown layer" / versi tidak cocok)**
Biasanya karena versi TensorFlow saat training (di Colab) berbeda dengan versi di laptop kamu. Cek versi TensorFlow Colab dengan `import tensorflow; print(tensorflow.__version__)` lalu samakan versi di `requirements.txt`.

**Error "feature_layer not found" (untuk CNN+SVM)**
Pastikan `best_CNN_extractor.h5` yang dipakai adalah model CNN LENGKAP (sampai layer output softmax), bukan yang sudah dipotong — kode di `app.py` akan memotongnya sendiri sampai layer `feature_layer`.

**Hasil prediksi terasa kurang akurat / beda dari saat training**
Pastikan urutan `CLASSES` di `app.py` (`['Sehat', 'Blight', 'Blast']`) sama persis dengan urutan saat training di notebook. Cek juga preprocessing gambar — fungsi `preprocess_image_raw()` di `app.py` mereplikasi median blur + histogram equalization seperti di notebook training; kalau notebook training kamu sudah diubah, sesuaikan juga fungsi ini.
