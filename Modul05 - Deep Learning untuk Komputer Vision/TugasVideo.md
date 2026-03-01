# TUGAS VIDEO MODUL 5: DEEP LEARNING UNTUK KOMPUTER VISION

---

## Deskripsi Tugas
Buat video laporan praktikum yang mendemonstrasikan seluruh materi, 10 percobaan, dan project deep learning. Tunjukkan pemahaman end-to-end: teori → implementasi → aplikasi.

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

### 3. Demo 10 Percobaan (30–40 menit)
Demonstrasikan setiap percobaan secara LIVE:

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | Klasifikasi dengan DNN OpenCV | Load model, preprocessing, top-5 prediksi |
| 2 | Perbandingan Model Pre-trained | Tabel akurasi + FPS per model |
| 3 | CNN dengan PyTorch | Arsitektur, training loop, loss curve |
| 4 | CNN dengan Keras | Arsitektur, training, evaluate |
| 5 | Transfer Learning | Freeze/unfreeze layers, perbandingan akurasi |
| 6 | Data Augmentation | Sebelum vs sesudah augmentasi, efek pada akurasi |
| 7 | Deteksi Objek YOLO | Loading model, deteksi pada gambar statik |
| 8 | Deteksi Objek Real-time | Demo webcam real-time dengan FPS |
| 9 | Segmentasi Semantik | Visualisasi overlay segmentasi penuh |
| 10 | Instance Segmentation + ONNX | Segmentasi per instance, export ONNX, re-inference |

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
| Durasi | 60–75 menit |
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
| Demo 10 Percobaan | 40% | Semua berjalan, penjelasan per langkah |
| Demo Project | 20% | Fitur lengkap, berjalan baik |
| Analisis & Kesimpulan | 10% | Kritis, perbandingan kuantitatif |
| Kualitas Video | 10% | Resolusi, audio, editing |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Demo di GPU / cloud training | +5 |
| Perbandingan kuantitatif detail (tabel, plot) | +5 |
| Video < 45 menit (tidak lengkap) | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul05`
