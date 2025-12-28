Sistem deteksi masker berbasis Deep Learning menggunakan arsitektur MobileNetV2 yang mampu mengklasifikasikan wajah ke dalam 3 kategori: With Mask, Without Mask, dan Incorrect Mask.

1. Struktur Folder: Pastikan folder app, src, config, data, dan outputs berada sejajar di dalam satu folder utama (Root).

2. File requirements.txt: Install Requierements.

3. Deployment: pada VS Code pilih terminal, lalu new terminal, setelah itu ketik 'streamlit run app/app.py --server.fileWatcherType none' pada terminal

4. Cara pakai: upload image/gambar wajah, Lalu klik "RUN ANALYSIS", hasil akan menunjukkan klasifikasi dari image dan seberapa confident model. untuk Real-time bisa langsung klik "start" (Pastikan Internet Stabil). 
