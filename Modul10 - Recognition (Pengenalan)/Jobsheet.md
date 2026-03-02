# JOBSHEET PRAKTIKUM
# MODUL 10: RECOGNITION (PENGENALAN)

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Mendeteksi wajah menggunakan Haar Cascade dan memahami konsep DNN Face Detection.
2. Mengenali wajah menggunakan LBPH Face Recognizer dan Eigenfaces (PCA).
3. Mendeteksi landmark wajah dan memahami 68-point model.
4. Melakukan preprocessing gambar untuk OCR (binarisasi, denoise, deskew).
5. Memahami pipeline OCR dengan Tesseract.
6. Mendeteksi teks pada scene menggunakan MSER dan morphology.
7. Mendeteksi pejalan kaki dengan HOG descriptor + SVM.
8. Mendeteksi kendaraan menggunakan contour analysis dan Haar Cascade.
9. Mengenali gestur tangan menggunakan convexity defects.
10. Mengklasifikasikan objek dengan Bag of Visual Words (BoVW).
11. Mengenali jenis scene berdasarkan histogram warna dan tekstur.
12. Memahami konsep face embedding dan perbandingan jarak.
13. Menghitung metrik evaluasi klasifikasi (Accuracy, Precision, Recall, F1).
14. Menghitung metrik evaluasi deteksi (IoU, mAP).
15. Menganalisis performa recognition dengan ROC curve dan AUC.
16. Melakukan tracking beberapa wajah pada video.
17. Membangun pipeline recognition lengkap: detect → align → embed → match.
18. Membuat proyek sistem pengenalan wajah dengan database dan evaluasi.

---

## 2. ALAT DAN BAHAN

### Perangkat Keras
- Laptop/PC (webcam opsional untuk tracking)

### Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `opencv-contrib-python`, `numpy`, `matplotlib`
- Opsional: `pytesseract`, `dlib`

### Dataset
- Gambar dari folder `image/` (diunduh via `download_image.py`)
- Dataset sintetis yang di-generate oleh program

---

## 3. LANGKAH KERJA

### Percobaan 1: Face Detection dengan Haar Cascade

**Tujuan**: Mendeteksi wajah menggunakan Haar Cascade Classifier bawaan OpenCV.

**Langkah Kerja**:
1. Buat file `01_face_detection_haar_cascade.py`.
2. Load gambar wajah dari folder `image/`.
3. Konversi ke grayscale, load `haarcascade_frontalface_default.xml`.
4. Deteksi wajah dengan `detectMultiScale(scaleFactor, minNeighbors, minSize)`.
5. Variasikan parameter: ubah scaleFactor (1.05, 1.1, 1.3) dan minNeighbors (3, 5, 8).
6. Coba cascade lain: `haarcascade_eye`, `haarcascade_smile`.
7. Gambar bounding box dan hitung jumlah wajah per variasi.
8. Simpan hasil ke folder `output/`.

**Analisis**: Pengaruh scaleFactor terhadap sensitivitas deteksi. Trade-off minNeighbors: terlalu rendah = banyak false positive, terlalu tinggi = miss deteksi.

---

### Percobaan 2: Face Detection DNN (Konsep)

**Tujuan**: Memahami konsep deteksi wajah berbasis Deep Neural Network dan perbandingannya dengan Haar.

**Langkah Kerja**:
1. Buat file `02_face_detection_dnn_konsep.py`.
2. Implementasikan deteksi DNN jika model tersedia, atau fallback ke Haar.
3. Bandingkan waktu inferensi Haar vs DNN (simulasi).
4. Tampilkan tabel perbandingan: metode, waktu, jumlah deteksi.
5. Simpan visualisasi ke folder `output/`.

**Analisis**: DNN lebih akurat (99%+ pada WIDER FACE) tetapi lebih lambat. Haar sangat cepat tetapi lebih banyak false positive.

---

### Percobaan 3: Face Recognition LBPH

**Tujuan**: Mengenali wajah menggunakan Local Binary Pattern Histogram (LBPH) Face Recognizer.

**Langkah Kerja**:
1. Buat file `03_face_recognition_lbph.py`.
2. Hitung LBP manual: bandingkan piksel center dengan 8 neighbor.
3. Buat dataset sintetis atau gunakan gambar dari `image/faces/`.
4. Train `cv2.face.LBPHFaceRecognizer_create()`.
5. Prediksi identitas wajah dan tampilkan confidence.
6. Simpan hasil ke folder `output/`.

**Analisis**: LBPH robust terhadap perubahan pencahayaan. Confidence rendah = lebih yakin.

---

### Percobaan 4: Face Recognition Eigenfaces

**Tujuan**: Mengenali wajah menggunakan Eigenfaces (PCA — Principal Component Analysis).

**Langkah Kerja**:
1. Buat file `04_face_recognition_eigenfaces.py`.
2. Hitung eigenfaces menggunakan PCA/SVD pada kumpulan wajah.
3. Visualisasikan 5 eigenfaces teratas.
4. Rekonstruksi wajah dengan komponen berbeda (5, 10, 20, 50).
5. Gunakan `cv2.face.EigenFaceRecognizer_create()` untuk recognition.
6. Simpan hasil ke folder `output/`.

**Analisis**: Eigenfaces menangkap variasi utama wajah. Semakin banyak komponen, rekonstruksi semakin akurat tetapi lebih rentan noise.

---

### Percobaan 5: Face Landmark Detection

**Tujuan**: Mendeteksi titik-titik landmark pada wajah (68-point model).

**Langkah Kerja**:
1. Buat file `05_face_landmark_detection.py`.
2. Deteksi wajah dengan Haar, estimasi posisi landmark berdasarkan proporsi.
3. Visualisasikan diagram 68 titik landmark standar.
4. Gambar titik: mata, hidung, mulut pada gambar wajah.
5. Simpan hasil ke folder `output/`.

**Analisis**: Landmark penting untuk face alignment, expression analysis, dan face morphing.

---

### Percobaan 6: OCR Preprocessing

**Tujuan**: Melakukan preprocessing gambar teks untuk meningkatkan akurasi OCR.

**Langkah Kerja**:
1. Buat file `06_ocr_preprocessing.py`.
2. Terapkan binarisasi: Otsu, Adaptive Mean, Adaptive Gaussian.
3. Terapkan denoising: Gaussian Blur, Median Blur, fastNlMeansDenoising.
4. Implementasikan deskew: hitung sudut kemiringan dari momen → rotasi.
5. Visualisasikan semua tahap preprocessing side-by-side.
6. Simpan hasil ke folder `output/`.

**Analisis**: Preprocessing yang tepat sangat meningkatkan akurasi OCR. Adaptive threshold lebih baik untuk pencahayaan tidak merata.

---

### Percobaan 7: OCR Tesseract (Konsep)

**Tujuan**: Memahami pipeline OCR: preprocessing → text detection → recognition.

**Langkah Kerja**:
1. Buat file `07_ocr_tesseract_konsep.py`.
2. Buat gambar berisi teks sintetis.
3. Deteksi baris teks menggunakan projection profile.
4. Jika `pytesseract` tersedia, lakukan actual OCR.
5. Visualisasikan pipeline OCR step-by-step.
6. Simpan hasil ke folder `output/`.

**Analisis**: Pipeline OCR: Input → Grayscale → Binarisasi → Denoise → Text Detection → Recognition.

---

### Percobaan 8: Scene Text Detection

**Tujuan**: Mendeteksi teks pada gambar pemandangan/scene menggunakan MSER dan morphology.

**Langkah Kerja**:
1. Buat file `08_scene_text_detection.py`.
2. Deteksi region teks menggunakan MSER (Maximally Stable Extremal Regions).
3. Deteksi teks menggunakan gradient + morphology (Sobel → dilasi → contour).
4. Filter bounding box berdasarkan aspect ratio.
5. Bandingkan kedua metode side-by-side.
6. Simpan hasil ke folder `output/`.

**Analisis**: MSER mendeteksi region stabil, morphology mengelompokkan karakter. Kedua metode cocok untuk scene text.

---

### Percobaan 9: Pedestrian Detection dengan HOG

**Tujuan**: Mendeteksi pejalan kaki menggunakan HOG descriptor + SVM bawaan OpenCV.

**Langkah Kerja**:
1. Buat file `09_pedestrian_detection_hog.py`.
2. Gunakan `cv2.HOGDescriptor` dengan `getDefaultPeopleDetector()`.
3. Deteksi pedestrian dengan `detectMultiScale`.
4. Visualisasikan gradient HOG (magnitude dan orientasi).
5. Gambar bounding box dengan confidence score.
6. Simpan hasil ke folder `output/`.

**Analisis**: HOG menangkap distribusi gradien terarah. Invariant terhadap perubahan pencahayaan global.

---

### Percobaan 10: Vehicle Detection

**Tujuan**: Mendeteksi kendaraan menggunakan contour analysis dan Haar Cascade.

**Langkah Kerja**:
1. Buat file `10_vehicle_detection.py`.
2. Deteksi kendaraan via edge detection + contour filtering (luas, aspect ratio).
3. Coba Haar Cascade `haarcascade_car.xml` jika tersedia.
4. Bandingkan kedua metode.
5. Simpan hasil ke folder `output/`.

**Analisis**: Contour-based sederhana tetapi terbatas. Haar cascade atau DNN lebih robust.

---

### Percobaan 11: Hand Gesture Recognition

**Tujuan**: Mengenali gestur tangan menggunakan convexity defects.

**Langkah Kerja**:
1. Buat file `11_hand_gesture_recognition.py`.
2. Segmentasi tangan menggunakan warna kulit pada HSV.
3. Hitung convex hull dan convexity defects pada kontur terbesar.
4. Hitung jumlah jari berdasarkan sudut antar defect.
5. Tampilkan kontur + hull + label jumlah jari.
6. Simpan hasil ke folder `output/`.

**Analisis**: Convexity defects menandai celah antar jari. Sudut < 90° = jari.

---

### Percobaan 12: Object Classification dengan Bag of Visual Words

**Tujuan**: Mengklasifikasikan objek menggunakan pendekatan BoVW dengan ORB descriptor.

**Langkah Kerja**:
1. Buat file `12_object_classification_bovw.py`.
2. Buat dataset sintetis 3 kelas: Lingkaran, Kotak, Segitiga.
3. Ekstrak fitur ORB dari setiap gambar.
4. Buat visual dictionary via K-Means clustering.
5. Hitung histogram BoVW per gambar.
6. Bandingkan histogram per kelas.
7. Simpan hasil ke folder `output/`.

**Analisis**: BoVW mengubah gambar menjadi histogram visual words yang bisa diklasifikasikan.

---

### Percobaan 13: Scene Recognition

**Tujuan**: Mengenali jenis scene/pemandangan berdasarkan histogram warna dan tekstur.

**Langkah Kerja**:
1. Buat file `13_scene_recognition.py`.
2. Ekstrak histogram warna RGB sebagai fitur.
3. Ekstrak fitur tekstur menggunakan LBP sederhana.
4. Klasifikasi scene berdasarkan jarak histogram.
5. Bandingkan histogram per jenis scene (Pantai, Kota, Hutan).
6. Simpan hasil ke folder `output/`.

**Analisis**: Scene yang berbeda memiliki distribusi warna dan tekstur yang khas.

---

### Percobaan 14: Face Embedding dan Distance

**Tujuan**: Memahami konsep face embedding dan perbandingan jarak.

**Langkah Kerja**:
1. Buat file `14_face_embedding_distance.py`.
2. Hitung embedding sederhana: resize → flatten → L2 normalize.
3. Hitung distance matrix: Euclidean dan Cosine.
4. Visualisasikan distance matrix sebagai heatmap.
5. Analisis: jarak kecil = wajah sama, jarak besar = wajah beda.
6. Simpan hasil ke folder `output/`.

**Analisis**: Embedding merepresentasikan wajah sebagai vektor. ArcFace/FaceNet menghasilkan 128/512-d vector.

---

### Percobaan 15: Evaluasi Classification Metrics

**Tujuan**: Menghitung dan memvisualisasikan metrik evaluasi klasifikasi.

**Langkah Kerja**:
1. Buat file `15_evaluasi_classification_metrics.py`.
2. Simulasikan prediksi classification (4 kelas, 200 sampel).
3. Hitung Confusion Matrix dari scratch.
4. Hitung Accuracy, Precision, Recall, F1 per kelas.
5. Visualisasikan sebagai heatmap + bar chart.
6. Simpan hasil ke folder `output/`.

**Analisis**: Accuracy bisa menyesatkan pada data imbalance. F1 lebih informatif.

---

### Percobaan 16: Evaluasi Detection Metrics

**Tujuan**: Menghitung metrik evaluasi deteksi: IoU dan Average Precision.

**Langkah Kerja**:
1. Buat file `16_evaluasi_detection_metrics.py`.
2. Implementasikan fungsi IoU dari scratch.
3. Simulasikan GT vs prediksi bounding boxes.
4. Hitung True Positive pada berbagai IoU threshold.
5. Plot TP vs IoU threshold.
6. Simpan hasil ke folder `output/`.

**Analisis**: IoU ≥ 0.5 = standard match. mAP@50 dan mAP@75 untuk evaluasi deteksi.

---

### Percobaan 17: ROC Curve dan AUC

**Tujuan**: Menganalisis performa recognition menggunakan ROC curve.

**Langkah Kerja**:
1. Buat file `17_recognition_roc_curve.py`.
2. Simulasikan skor genuine (same person) dan impostor (different person).
3. Hitung TPR dan FPR pada berbagai threshold.
4. Plot ROC curve dan hitung AUC.
5. Bandingkan model baik, sedang, dan buruk.
6. Tampilkan distribusi skor genuine vs impostor.
7. Simpan hasil ke folder `output/`.

**Analisis**: AUC = 1.0 ideal. Distribusi skor yang terpisah baik = recognition akurat.

---

### Percobaan 18: Multi Face Tracking

**Tujuan**: Tracking beberapa wajah pada video/frame berurutan.

**Langkah Kerja**:
1. Buat file `18_multi_face_tracking.py`.
2. Simulasikan 5 frame dengan wajah bergerak.
3. Implementasikan tracking sederhana: matching center distance.
4. Assign ID per wajah, pertahankan antar frame.
5. Visualisasikan tracking dengan warna per ID.
6. Simpan hasil ke folder `output/`.

**Analisis**: Tracking = deteksi + assignment ID. Tantangan: oklusi, entry/exit wajah baru.

---

### Percobaan 19: Recognition Pipeline Lengkap

**Tujuan**: Membangun pipeline recognition end-to-end.

**Langkah Kerja**:
1. Buat file `19_recognition_pipeline_lengkap.py`.
2. Step 1: Detect wajah (Haar/DNN).
3. Step 2: Align (crop + resize + equalize histogram).
4. Step 3: Extract embedding (resize → flatten → normalize).
5. Step 4: Match dengan database (Euclidean distance + threshold).
6. Visualisasikan pipeline diagram.
7. Simpan hasil ke folder `output/`.

**Analisis**: Pipeline standard: Detect → Align → Embed → Match → Result.

---

### Percobaan 20: Proyek Recognition Sistem

**Tujuan**: Membangun sistem pengenalan wajah lengkap dengan database dan evaluasi.

**Langkah Kerja**:
1. Buat file `20_proyek_recognition_sistem.py`.
2. Buat database wajah dari folder `image/faces/`.
3. Train LBPH recognizer dari database.
4. Evaluasi: confusion matrix, per-class accuracy.
5. Tampilkan sampel database dan hasil evaluasi.
6. Simpan hasil ke folder `output/`.

**Analisis**: Integrasi semua konsep modul: deteksi, recognition, embedding, evaluasi.

---

## 4. ANALISIS DAN PEMBAHASAN

### Pertanyaan Analisis
1. Bandingkan waktu dan akurasi Haar Cascade vs DNN untuk face detection.
2. Kapan LBPH lebih baik dari Eigenfaces? Analisis kelebihan masing-masing.
3. Mengapa preprocessing sangat penting untuk akurasi OCR?
4. Jelaskan hubungan antara IoU threshold dan jumlah True Positive.
5. Bagaimana face embedding mengubah masalah recognition menjadi distance computation?

---

## 5. KESIMPULAN

Tulis kesimpulan yang mencakup:
- Perbandingan metode deteksi wajah (Haar vs DNN).
- Perbandingan metode recognition (LBPH vs Eigenfaces vs Embedding).
- Pentingnya preprocessing untuk OCR.
- Metrik evaluasi yang tepat untuk classification vs detection.
- Pipeline recognition end-to-end.
