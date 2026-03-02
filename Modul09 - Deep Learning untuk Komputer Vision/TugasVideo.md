# TUGAS VIDEO MODUL 9: DEEP LEARNING UNTUK KOMPUTER VISION

---

## Deskripsi Tugas
Buat video laporan praktikum yang mendemonstrasikan seluruh materi, 20 percobaan, dan project deep learning. Tunjukkan pemahaman end-to-end: teori → implementasi → aplikasi.

---

## Struktur Video

### 1. Pembukaan (Maks 2 menit)
- Perkenalan: Nama, NIM, kelas, modul.
- Overview singkat topik Deep Learning untuk Computer Vision.

### 2. Penjelasan Materi (10–15 menit)
- Arsitektur Neural Network dan CNN.
- Konsep convolution, pooling, activation function.
- Arsitektur terkenal: LeNet → AlexNet → VGG → ResNet → EfficientNet.
- Transfer learning dan fine-tuning.
- Data augmentation: teknik dan manfaat.
- Object detection: sliding window, HOG, YOLO.
- Semantic vs instance segmentation.
- Loss functions dan optimizers.
- Batch Normalization, Dropout.
- Deployment: ONNX, quantization.
- **Wajib**: Gambar/diagram arsitektur CNN + tunjukkan di layar.

### 3. Demo 20 Percobaan (40–60 menit)
Demonstrasikan setiap percobaan secara LIVE:

| No | Percobaan | Poin Penting |
|----|-----------|---------------|
| 1 | OpenCV DNN Blob | blobFromImage, parameter preprocessing, visualisasi channel |
| 2 | Klasifikasi DNN | readNet, forward, top-5 prediksi, confidence |
| 3 | Perbandingan Model | Radar chart 5 model, trade-off akurasi vs kecepatan |
| 4 | Arsitektur CNN Visualisasi | Conv→ReLU→Pool pipeline, feature map per layer |
| 5 | Fungsi Aktivasi | ReLU/Sigmoid/Tanh/LeakyReLU plot dan perbandingan |
| 6 | Konvolusi dan Pooling | Kernel Sobel/Gaussian, MaxPool vs AvgPool |
| 7 | Data Augmentasi Dasar | 8 teknik augmentasi, grid sebelum/sesudah |
| 8 | Transfer Learning Konsep | 3 strategi, training curves simulasi |
| 9 | Backpropagation Visualisasi | XOR problem, loss curve, decision boundary |
| 10 | Deteksi Sliding Window | Window scan, image pyramid, NMS |
| 11 | Deteksi HOG | Gradient, HOG descriptor, pedestrian detection |
| 12 | YOLO Konsep Grid | Grid overlay, IoU, anchor boxes |
| 13 | Semantic Segmentation Manual | HSV segmentasi, K-Means, overlay |
| 14 | Instance Segmentation Konsep | Per-instance mask, 4-task comparison |
| 15 | Loss Function Visualisasi | MSE/BCE/CCE/Dice/Focal plot |
| 16 | Optimizer Visualisasi | SGD/Momentum/Adam/RMSProp trajectory |
| 17 | Batch Normalization & Dropout | Distribusi aktivasi, efek regularisasi |
| 18 | Model Evaluasi Metrik | Confusion matrix, ROC, PR curve |
| 19 | ONNX dan Deployment | Pipeline diagram, quantization, benchmark |
| 20 | Proyek Klasifikasi Bentuk | End-to-end: dataset→fitur→KNN→evaluasi |

- Tunjukkan kode berjalan di IDE / terminal.
- Jelaskan setiap langkah dan hasilnya.

### 4. Demo Project (10–15 menit)
- Tunjukkan project yang dipilih dari Soal Cerita.
- Demo fitur lengkap dari awal sampai output.
- Jelaskan arsitektur, dataset, training, dan hasilnya.
- Tunjukkan metrik (akurasi, precision, recall, dll.).

### 5. Analisis dan Penutup (5 menit)
- Rangkum temuan penting.
- Bandingkan metode/arsitektur yang dicoba.
- Tantangan yang dihadapi dan solusinya.
- Kesimpulan dan saran.

---

## Ketentuan Teknis

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 75–100 menit |
| Resolusi | Minimal 1080p |
| Recording | Screen recording + webcam (picture-in-picture) |
| Webcam | Wajah terlihat jelas saat menjelaskan |
| Audio | Narasi jelas, tanpa background noise berlebihan |
| Platform | Upload ke YouTube (Unlisted) atau Google Drive |
| Format | MP4 |

---

## Rubrik Penilaian Video

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Pembukaan | 5% | Profesional, lengkap |
| Penjelasan Materi | 15% | Akurat, mendalam, diagram jelas |
| Demo 20 Percobaan | 40% | Semua berjalan, penjelasan per langkah |
| Demo Project | 20% | Fitur lengkap, berjalan baik |
| Analisis & Kesimpulan | 10% | Kritis, perbandingan kuantitatif |
| Kualitas Video | 10% | Resolusi, audio, editing |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Demo di GPU / cloud training | +5 |
| Perbandingan kuantitatif detail (tabel, plot) | +5 |
| Video < 60 menit (tidak lengkap) | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul09`
