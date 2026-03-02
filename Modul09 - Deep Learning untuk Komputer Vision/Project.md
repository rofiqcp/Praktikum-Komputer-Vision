# PROJECT MODUL 9: DEEP LEARNING UNTUK KOMPUTER VISION

---

## Deskripsi Umum
Project mengintegrasikan konsep deep learning dari 20 percobaan: preprocessing DNN (blob), klasifikasi (OpenCV DNN), arsitektur CNN, fungsi aktivasi, konvolusi & pooling, augmentasi data, transfer learning, backpropagation, deteksi objek (sliding window, HOG, YOLO), segmentasi (semantic, instance), loss functions, optimizers, Batch Normalization & Dropout, metrik evaluasi, dan deployment ONNX. Pilih minimal 1 soal cerita.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Custom Object Detector** — Train YOLOv8 pada dataset custom (5 kelas, 100+ gambar).
2. **Real-time Face Mask Detector** — Fine-tune model untuk mendeteksi masker wajah.
3. **Plant Disease Classifier** — Transfer learning untuk klasifikasi penyakit tanaman dari foto daun.
4. **Multi-model Ensemble** — Gabungkan prediksi beberapa model untuk akurasi lebih tinggi.
5. **GradCAM Visualizer** — Visualisasikan area gambar yang paling berpengaruh pada prediksi CNN.
6. **Augmentation Policy Searcher** — Otomatis cari kombinasi augmentasi terbaik.
7. **Model Pruning Benchmark** — Pruning model dan ukur trade-off akurasi vs kecepatan.
8. **Edge Deployment** — Deploy model ke Raspberry Pi / Jetson Nano menggunakan ONNX.
9. **Semi-supervised Learning** — Manfaatkan data tanpa label untuk meningkatkan model.
10. **Semantic Segmentation Fine-tuner** — Fine-tune DeepLab untuk segmentasi domain spesifik.
11. **Video Object Detection Pipeline** — Deteksi + tracking objek pada video panjang.
12. **Mobile-optimized Classifier** — MobileNet + quantization untuk deployment mobile.
13. **Few-shot Learning Demo** — Klasifikasi kelas baru dengan hanya 5 contoh.
14. **Auto-annotation Tool** — Gunakan model pre-trained untuk auto-generate anotasi.
15. **Model Explainability Dashboard** — GradCAM + confusion matrix + per-class metrics.

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Sistem Sortir Sampah Otomatis
Pemerintah kota ingin memasang mesin sortir sampah di tempat umum yang menggunakan kamera untuk mengklasifikasikan jenis sampah (organik, plastik, kertas, logam, kaca). Buatlah: (a) dataset 5 kelas (kumpulkan 50+ gambar per kelas dari internet), (b) train classifier (transfer learning), (c) data augmentation, (d) deteksi objek sampah (YOLO/sliding window), (e) real-time classification dari webcam, (f) statistik jumlah per jenis, (g) export model ke ONNX.

### Soal 2: Asisten Pengemudi (Driver Assistance)
Startup otomotif memerlukan prototipe asisten pengemudi. Buatlah: (a) deteksi kendaraan + pejalan kaki (YOLO/HOG), (b) segmentasi jalan (semantic segmentation), (c) estimasi jarak berdasarkan ukuran bounding box, (d) alert jika pejalan kaki terlalu dekat, (e) processing dari video dashcam, (f) logging event berbahaya, (g) statistik per trip.

### Soal 3: Sistem Monitoring Kehadiran Kelas
Kampus ingin memantau kehadiran otomatis dari CCTV kelas. Buatlah: (a) deteksi orang (YOLO/HOG), (b) counting jumlah mahasiswa per frame, (c) instance segmentation untuk memisahkan tiap orang, (d) grafik kehadiran per waktu, (e) deteksi posisi duduk (kursi terisi/kosong), (f) real-time dari webcam, (g) export laporan.

### Soal 4: Aplikasi Identifikasi Spesies Tanaman
Kebun raya memerlukan aplikasi untuk pengunjung mengidentifikasi tanaman dari foto. Buatlah: (a) dataset 10 spesies tanaman, (b) transfer learning (MobileNet/ResNet), (c) augmentasi fotografi outdoor, (d) visualisasi area identifikasi (GradCAM/heatmap), (e) top-3 prediksi dengan confidence, (f) informasi per spesies, (g) interface sederhana.

### Soal 5: Quality Control Defect Detection
Pabrik elektronik memerlukan deteksi cacat pada PCB. Buatlah: (a) dataset gambar PCB normal dan cacat, (b) train model deteksi cacat, (c) segmentasi area cacat, (d) klasifikasi jenis cacat (scratch, missing component, misalignment), (e) confidence scoring, (f) heatmap area cacat, (g) pass/fail decision.

### Soal 6: Sistem Pemantau Hewan di Kebun Binatang
Kebun binatang ingin monitoring otomatis aktivitas hewan. Buatlah: (a) deteksi hewan dari kamera CCTV (YOLO custom), (b) klasifikasi spesies, (c) tracking posisi, (d) analisis aktivitas (bergerak/diam), (e) alert jika hewan tidak terdeteksi > 1 jam, (f) statistik aktivitas harian, (g) dashboard visual.

### Soal 7: Sistem Pembaca Buku untuk Tunanetra
Organisasi disabilitas memerlukan alat bantu membaca. Buatlah: (a) deteksi area teks dalam gambar, (b) segmentasi teks dari background, (c) preprocessing untuk OCR, (d) integrasi Tesseract OCR, (e) text-to-speech (opsional), (f) support layout buku, (g) batch processing halaman.

### Soal 8: Analisis Citra Satelit untuk Urban Planning
Pemerintah daerah ingin menganalisis perkembangan area urban. Buatlah: (a) segmentasi semantik citra satelit (bangunan, jalan, vegetasi, air), (b) hitung persentase area per kelas, (c) perbandingan temporal (2 gambar tahun berbeda), (d) deteksi bangunan baru, (e) visualisasi change map, (f) statistik perkembangan, (g) export report.

### Soal 9: Fitness Repetition Counter
Pusat kebugaran ingin tool penghitung repetisi otomatis. Buatlah: (a) deteksi orang (YOLO/HOG), (b) pose estimation atau bounding box tracking, (c) analisis gerakan (naik/turun), (d) counting repetisi, (e) real-time dari webcam, (f) timer dan rest counter, (g) statistik per sesi.

### Soal 10: Fashion Item Classifier dan Recommender
E-commerce fashion memerlukan tool klasifikasi produk. Buatlah: (a) klasifikasi jenis pakaian (10 kategori), (b) transfer learning, (c) segmentasi pakaian dari background, (d) ekstraksi warna dominan, (e) rekomendasi kemiripan fitur, (f) interface upload foto, (g) top-5 rekomendasi.

---

## Rubrik Penilaian Project

| Komponen | Bobot |
|----------|-------|
| Fungsionalitas | 35% |
| Integrasi Percobaan | 20% |
| Kualitas Kode | 15% |
| Dokumentasi | 15% |
| Kreativitas | 15% |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project09.zip`
- **Catatan**: Sertakan model weights atau instruksi download-nya.
