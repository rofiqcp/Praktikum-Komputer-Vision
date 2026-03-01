# TUGAS VIDEO MODUL 5: DEEP LEARNING UNTUK KOMPUTER VISION

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
- Object detection (YOLO): arsitektur + pipeline.
- Semantic vs instance segmentation.
- Deployment: ONNX, quantization.
- **Wajib**: Gambar/diagram arsitektur CNN + tunjukkan di layar.

### 3. Demo 20 Percobaan (40–60 menit)
Demonstrasikan setiap percobaan secara LIVE:

| No | Percobaan | Poin Penting |
|----|-----------|---------------|
| 1 | Klasifikasi dengan DNN OpenCV Blob | Load model, preprocessing, blob creation, top-5 prediksi |
| 2 | Klasifikasi dengan OpenCV DNN | Inferensi DNN, parsing output, label mapping |
| 3 | Perbandingan Model Pre-trained | Tabel akurasi + FPS per model |
| 4 | Visualisasi Arsitektur CNN | Layer-by-layer, feature map visualization |
| 5 | Fungsi Aktivasi | ReLU, Sigmoid, Tanh — plot dan perbandingan |
| 6 | Operasi Konvolusi dan Pooling | Filter, stride, padding, pooling demo |
| 7 | Data Augmentasi Dasar | Sebelum vs sesudah augmentasi, variasi teknik |
| 8 | Transfer Learning (Konsep) | Freeze/unfreeze layers, perbandingan akurasi |
| 9 | Backpropagation Visualisasi | Gradient flow, update weight, loss descent |
| 10 | Deteksi Objek Sliding Window | Window scanning, multi-scale, NMS |
| 11 | Deteksi Objek dengan HOG | HOG descriptor, people detection, parameter tuning |
| 12 | YOLO Konsep Grid | Grid overlay, anchor boxes, IoU, NMS |
| 13 | Semantic Segmentation Manual | Thresholding, color-based, morphology, overlay |
| 14 | Instance Segmentation (Konsep) | Connected components, watershed, per-instance mask |
| 15 | Loss Function Visualisasi | MSE, Cross-Entropy, Dice, Focal Loss plot |
| 16 | Optimizer Visualisasi | SGD vs SGDM vs Adam trajectory, convergence |
| 17 | Batch Normalization & Dropout | Distribusi aktivasi, efek regularisasi |
| 18 | Model Evaluasi Metrik | Confusion matrix, PR curve, ROC, mAP |
| 19 | ONNX dan Deployment | Export, runtime inferensi, quantization |
| 20 | Proyek Klasifikasi Bentuk | End-to-end: dataset → training → evaluasi → ONNX |

- Tunjukkan kode berjalan di IDE / Jupyter Notebook.
- Jelaskan setiap langkah dan hasilnya.

### 4. Demo Project (10–15 menit)
- Tunjukkan project yang dipilih dari Soal Cerita.
- Demo fitur lengkap dari awal sampai output.
- Jelaskan arsitektur, dataset, training, dan hasilnya.
- Tunjukkan metrik (akurasi, mAP, IoU, dll.).

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
- **Naming**: `[NIM]_[Nama]_Video_Modul05`
