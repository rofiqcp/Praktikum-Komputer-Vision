# TUGAS VIDEO MODUL 6: RECOGNITION (PENGENALAN)

---

## Deskripsi Tugas
Buat video laporan praktikum yang mendemonstrasikan seluruh materi, 20 percobaan, dan project recognition. Video harus menunjukkan pemahaman mendalam tentang berbagai teknik recognition: face detection, face recognition, OCR, object classification, gesture recognition, tracking, pipeline integration, dan evaluation.

---

## Struktur Video

### 1. Pembukaan (Maks 2 menit)
- Perkenalan: Nama, NIM, kelas, modul.
- Overview singkat topik Recognition.

### 2. Penjelasan Materi (10–15 menit)
- Taxonomi recognition: instance vs category, classification vs detection vs segmentation.
- Face detection: Viola-Jones vs Deep Learning — arsitektur + perbedaan.
- Face recognition: Eigenfaces vs LBPH vs FaceNet — pipeline + embedding.
- Object classification: BoVW vs Deep Learning.
- OCR pipeline: preprocessing → detection → recognition.
- Pedestrian detection: HOG + SVM, konsep sliding window.
- Gesture recognition: MediaPipe landmarks.
- Metrics: accuracy, precision, recall, F1, IoU, mAP, ROC.
- **Wajib**: Diagram/slide untuk setiap topik utama.

### 3. Demo 20 Percobaan (60–80 menit)

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | Face Detection — Haar Cascade | Parameter tuning, side-by-side comparison |
| 2 | Face Detection — DNN | Confidence threshold, DNN vs Haar comparison |
| 3 | Face Recognition — LBPH | Training, prediction, confidence score |
| 4 | Face Recognition — Eigenfaces | PCA projection, eigenface visualization |
| 5 | Face Landmark Detection | 68 landmarks, facial feature extraction |
| 6 | OCR Preprocessing | Binarization, denoising, deskewing comparison |
| 7 | OCR Tesseract | Tesseract recognition, multi-language |
| 8 | Scene Text Detection | EAST detection + OCR pipeline end-to-end |
| 9 | Pedestrian Detection — HOG | Parameter tuning, sliding window demo |
| 10 | Vehicle Detection | Vehicle detection, counting, classification |
| 11 | Hand Gesture Recognition | MediaPipe landmarks, finger counting demo |
| 12 | Object Classification — BoVW | SIFT → KMeans → histogram → SVM pipeline |
| 13 | Scene Recognition | Spatial pyramid matching, scene classification |
| 14 | Face Embedding Distance | Euclidean vs Cosine, verification/identification |
| 15 | Classification Metrics | Accuracy, precision, recall, F1, confusion matrix |
| 16 | Detection Metrics | IoU, mAP, precision-recall curve |
| 17 | Recognition ROC Curve | ROC, AUC, EER, threshold tuning |
| 18 | Multi-Face Tracking | Detection → tracking, ID assignment |
| 19 | Recognition Pipeline Lengkap | End-to-end: detection + recognition + evaluation |
| 20 | Proyek Recognition Sistem | Complete recognition system demo |

- Jalankan kode LIVE di IDE / Jupyter Notebook.
- Jelaskan setiap langkah dan hasilnya.
- Tunjukkan demo real-time jika ada (face detection, gesture).

### 4. Demo Project (10–15 menit)
- Tunjukkan project yang dipilih dari Soal Cerita.
- Demo fitur lengkap dari awal sampai output.
- Jelaskan arsitektur, dataset, dan pipeline.
- Tunjukkan metrik evaluasi.

### 5. Analisis dan Penutup (5 menit)
- Rangkum temuan penting.
- Bandingkan metode klasik vs deep learning.
- Tantangan dan solusi.
- Kesimpulan dan saran pengembangan.

---

## Ketentuan Teknis

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 90–120 menit |
| Resolusi | Minimal 1080p |
| Recording | Screen recording + webcam (picture-in-picture) |
| Webcam | Wajah terlihat jelas |
| Audio | Narasi jelas, tanpa background noise berlebihan |
| Platform | YouTube (Unlisted) atau Google Drive |
| Format | MP4 |

---

## Rubrik Penilaian Video

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Pembukaan | 5% | Profesional, lengkap |
| Penjelasan Materi | 15% | Akurat, mendalam, diagram jelas |
| Demo 20 Percobaan | 40% | Semua berjalan, penjelasan per langkah |
| Demo Project | 20% | Fitur lengkap, berjalan baik |
| Analisis & Kesimpulan | 10% | Kritis, kuantitatif |
| Kualitas Video | 10% | Resolusi, audio, editing |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Demo real-time (webcam) untuk ≥3 percobaan | +5 |
| Perbandingan tabel akurasi lengkap | +5 |
| Video < 60 menit (tidak lengkap) | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul06`
