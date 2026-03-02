# TUGAS VIDEO — MODUL 10: RECOGNITION (PENGENALAN)

---

## Deskripsi Tugas
Mahasiswa membuat **video presentasi** yang membahas materi, percobaan, dan project Modul 10: Recognition (Pengenalan).

---

## Struktur Video

### Bagian 1: Pembukaan (1–2 menit)
- Perkenalan (nama, NIM, kelas)
- Gambaran umum Modul 10: Teknik-teknik pengenalan visual — dari face detection klasik hingga OCR, gesture recognition, dan metrik evaluasi

### Bagian 2: Penjelasan Materi (5–8 menit)
Pilih **minimal 5 topik** untuk dijelaskan:
1. Face Detection: Haar Cascade (integral image, AdaBoost, cascade) vs DNN
2. Face Recognition: Eigenfaces (PCA), LBPH, Face Embedding (FaceNet/ArcFace)
3. OCR Pipeline: Preprocessing → Text Detection → Character Recognition (Tesseract)
4. HOG Pedestrian Detection: Gradient histogram → cell → block → SVM
5. Bag of Visual Words (BoVW): Ekstrak fitur → K-Means → Histogram → Klasifikasi
6. Metrik Evaluasi: Confusion Matrix, Precision/Recall/F1, IoU, mAP, ROC-AUC
7. Recognition Pipeline: Detect → Align → Embed → Match → Result

### Bagian 3: Demo Percobaan (5–8 menit)
Jalankan dan jelaskan **minimal 5 percobaan** dari 20:

| No | File | Konsep Utama |
|----|------|-------------|
| 1 | `01_face_detection_haar_cascade.py` | Haar Cascade, scaleFactor, minNeighbors |
| 2 | `02_face_detection_dnn_konsep.py` | DNN face detection, perbandingan Haar vs DNN |
| 3 | `03_face_recognition_lbph.py` | LBP manual, LBPH recognizer, training dataset |
| 4 | `04_face_recognition_eigenfaces.py` | PCA/SVD, eigenfaces, rekonstruksi wajah |
| 5 | `05_face_landmark_detection.py` | Landmark wajah, 68-point model |
| 6 | `06_ocr_preprocessing.py` | Binarisasi, denoise, deskew untuk OCR |
| 7 | `07_ocr_tesseract_konsep.py` | Pipeline OCR, Tesseract integration |
| 8 | `08_scene_text_detection.py` | MSER, morphology-based text detection |
| 9 | `09_pedestrian_detection_hog.py` | HOG descriptor, gradient visualization |
| 10 | `10_vehicle_detection.py` | Contour-based + Haar cascade vehicle detection |
| 11 | `11_hand_gesture_recognition.py` | HSV segmentasi, convexity defects, finger count |
| 12 | `12_object_classification_bovw.py` | ORB features, K-Means, BoVW histogram |
| 13 | `13_scene_recognition.py` | Color histogram, LBP texture, scene klasifikasi |
| 14 | `14_face_embedding_distance.py` | Embedding sederhana, distance matrix, cosine sim |
| 15 | `15_evaluasi_classification_metrics.py` | Confusion matrix, P/R/F1, bar chart |
| 16 | `16_evaluasi_detection_metrics.py` | IoU, TP vs threshold, mAP konsep |
| 17 | `17_recognition_roc_curve.py` | ROC curve, AUC, score distribution |
| 18 | `18_multi_face_tracking.py` | Face tracking, ID assignment, multi-frame |
| 19 | `19_recognition_pipeline_lengkap.py` | Pipeline end-to-end: Detect→Align→Embed→Match |
| 20 | `20_proyek_recognition_sistem.py` | Sistem recognition: database, training, evaluasi |

### Bagian 4: Demo Project (3–5 menit)
- Jelaskan soal cerita yang dipilih
- Demo jalankan project
- Tunjukkan fitur utama dan output

### Bagian 5: Penutup (1–2 menit)
- Rangkuman materi dan percobaan
- Tantangan dan solusi yang ditemui
- Kesimpulan dan pembelajaran

---

## Ketentuan Teknis
- **Durasi**: 15–25 menit
- **Format**: MP4, resolusi minimal 720p
- **Nama file**: `Video_Modul10_NIM_Nama.mp4`
- **Platform**: Upload ke Google Drive / YouTube (unlisted)
