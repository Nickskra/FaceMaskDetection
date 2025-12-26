# Face Mask Detection Web

Aplikasi berbasis web untuk mendeteksi penggunaan masker secara real-time menggunakan Deep Learning (TensorFlow/Keras) dan Streamlit.

## Fitur
- **Upload Image**: Deteksi masker dari foto yang diunggah (JPG/PNG).
- **Real-Time Monitoring**: Deteksi langsung melalui webcam menggunakan WebRTC.
- **High Accuracy**: Menggunakan model Deep Learning yang sudah dilatih sebelumnya.

## Teknologi
- **Python** (Bahasa pemrograman utama)
- **TensorFlow/Keras** (Model Deep Learning)
- **Streamlit** (Web Interface)
- **OpenCV** (Pengolahan citra)

## Struktur Folder
- `app/app.py`: Kode utama aplikasi.
- `outputs/models/`: Lokasi penyimpanan model `.h5`.
- `requirements.txt`: Daftar library yang dibutuhkan.

## Cara buka
- **VS CODE**: buka app/README.md untuk lebih lengkapnya
- **Streamlit Cloud**: https://facemaskdetection-dl.streamlit.app/

## Cara pakai
- **Upload Image**: klik "Browse files", pilih image(jpg,jpeg,png), klik "RUN ANALYSIS"
- **Real-Time**: klik start(Pastikan Internet aman)
