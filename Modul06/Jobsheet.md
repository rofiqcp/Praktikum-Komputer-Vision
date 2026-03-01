# JOBSHEET MODUL 6: RECOGNITION (PENGENALAN)

---

## Tujuan Praktikum
1. Memahami dan mengimplementasikan face detection (Haar Cascade dan DNN).
2. Mengimplementasikan face recognition (LBPH, FaceNet/DeepFace).
3. Mengimplementasikan object recognition dan classification.
4. Melakukan OCR (Optical Character Recognition) pada berbagai tipe citra.
5. Mengimplementasikan pedestrian dan vehicle detection.
6. Mengimplementasikan hand gesture recognition.
7. Memahami evaluation metrics untuk recognition tasks.

---

## Alat dan Bahan
- **Hardware**: PC/Laptop, webcam.
- **Software**: Python 3.8+, Jupyter Notebook / VS Code.
- **Library**: OpenCV (`opencv-contrib-python`), NumPy, Matplotlib, Tesseract OCR, pytesseract, mediapipe, deepface, scikit-learn.
- **Dataset**: Gambar wajah (LFW subset atau koleksi sendiri), gambar teks, gambar scene.
- **Model**: Haar cascade, Caffe face detector, EAST text detector, YOLO weights.

### Instalasi Tambahan
```bash
pip install opencv-contrib-python numpy matplotlib pytesseract mediapipe deepface scikit-learn
# Install Tesseract OCR engine dari https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Percobaan 1: Face Detection — Haar Cascade Classifier

### Tujuan
Mendeteksi wajah menggunakan metode klasik Viola-Jones (Haar Cascade).

### Dasar Teori
Haar Cascade menggunakan fitur Haar-like yang dihitung cepat dengan integral image, dipilih oleh AdaBoost, dan disusun dalam cascade untuk reject cepat.

### Langkah Kerja
1. Load gambar yang berisi satu atau lebih wajah manusia.
2. Konversi ke grayscale.
3. Load `haarcascade_frontalface_default.xml` dari OpenCV.
4. Jalankan `detectMultiScale()` dengan parameter default.
5. Gambar bounding box pada wajah terdeteksi.
6. Variasikan parameter `scaleFactor` (1.05, 1.1, 1.3) dan amati perbedaan.
7. Variasikan `minNeighbors` (3, 5, 7) dan amati perbedaan.
8. Uji pada gambar dengan berbagai ukuran wajah.
9. Uji pada gambar grup (banyak wajah).
10. Tampilkan semua hasil side-by-side.

### Analisis Percobaan 1
- Berapa wajah terdeteksi dengan parameter berbeda?
- Pada `scaleFactor` berapa false positive mulai muncul?
- Apakah wajah miring/profil terdeteksi? Mengapa?
- Bagaimana `minNeighbors` mempengaruhi false positive dan false negative?

---

## Percobaan 2: Face Detection — Deep Learning (DNN)

### Tujuan
Mendeteksi wajah menggunakan model deep learning yang lebih robust daripada Haar Cascade.

### Dasar Teori
Model DNN (Caffe/TensorFlow SSD) menghasilkan deteksi yang lebih robust terhadap variasi pose, pencahayaan, dan oklusi dibandingkan Haar Cascade.

### Langkah Kerja
1. Download model Caffe face detector (`res10_300x300_ssd_iter_140000.caffemodel` + `deploy.prototxt`).
2. Load model dengan `cv2.dnn.readNetFromCaffe()`.
3. Preprocessing: buat blob dari gambar (`blobFromImage`).
4. Forward pass dan ambil deteksi.
5. Filter deteksi berdasarkan confidence threshold (coba 0.5, 0.7, 0.9).
6. Gambar bounding box + confidence percentage.
7. Bandingkan hasil dengan Haar Cascade pada gambar yang sama.
8. Uji pada gambar dengan pencahayaan rendah.
9. Uji pada gambar dengan wajah tertutup sebagian (masker/kacamata).
10. Ukur dan bandingkan waktu eksekusi DNN vs Haar Cascade.

### Analisis Percobaan 2
- Berapa confidence rata-rata deteksi wajah?
- Pada kondisi apa DNN lebih unggul dari Haar Cascade?
- Apakah wajah dengan masker terdeteksi? Bandingkan kedua metode.
- Berapa perbedaan waktu eksekusi?

---

## Percobaan 3: Face Recognition (LBPH)

### Tujuan
Mengimplementasikan face recognition menggunakan metode Local Binary Pattern Histogram.

### Dasar Teori
LBPH menghitung histogram dari pola biner lokal pada gambar wajah. Setiap wajah direpresentasikan sebagai histogram, lalu dibandingkan dengan database menggunakan jarak histogram.

### Langkah Kerja
1. Kumpulkan dataset wajah: minimal 3 orang, 10 foto per orang.
2. Crop dan resize wajah ke ukuran seragam (100×100 grayscale).
3. Bagi dataset: 70% training, 30% testing.
4. Buat dan train `cv2.face.LBPHFaceRecognizer`.
5. Predict wajah pada test set.
6. Hitung akurasi recognition.
7. Visualisasikan confidence score per prediksi.
8. Uji dengan foto baru yang tidak ada di dataset.
9. Uji dengan variasi ekspresi dan pencahayaan.
10. Tampilkan confusion matrix.

### Analisis Percobaan 3
- Berapa akurasi recognition pada test set?
- Pada confidence berapa recognizer mulai salah?
- Apakah variasi ekspresi mempengaruhi akurasi? Bagaimana?
- Bagaimana kualitas gambar mempengaruhi recognition?

---

## Percobaan 4: Face Recognition (DeepFace)

### Tujuan
Mengimplementasikan face recognition modern menggunakan library DeepFace.

### Dasar Teori
DeepFace menyediakan interface ke berbagai model face recognition (VGG-Face, FaceNet, ArcFace). Model menghasilkan embedding vektor dari wajah, lalu menghitung kemiripan menggunakan jarak Euclidean atau cosine.

### Langkah Kerja
1. Install library `deepface`.
2. Load dua gambar wajah orang yang sama → verifikasi.
3. Load dua gambar wajah orang berbeda → verifikasi.
4. Bandingkan hasil verifikasi dari 3 model berbeda (VGG-Face, Facenet, ArcFace).
5. Lakukan face analysis: umur, gender, emosi, ras.
6. Buat database wajah (folder berisi subfolder per orang).
7. Jalankan `find()` untuk mencari wajah serupa di database.
8. Plot embedding di 2D menggunakan t-SNE/PCA.
9. Hitung dan plot perbandingan jarak intra-class vs inter-class.
10. Bandingkan akurasi DeepFace vs LBPH pada dataset yang sama.

### Analisis Percobaan 4
- Model mana (VGG-Face, Facenet, ArcFace) yang paling akurat?
- Berapa threshold jarak optimal untuk verifikasi?
- Apakah analisis umur/gender/emosi akurat? Pada kasus apa?
- Seberapa besar peningkatan akurasi DeepFace dibandingkan LBPH?

---

## Percobaan 5: Object Classification — Bag of Visual Words

### Tujuan
Mengimplementasikan object classification klasik menggunakan pendekatan Bag of Visual Words.

### Dasar Teori
BoVW menganalogikan pengolahan gambar dengan pengolahan teks — keypoint descriptor dikelompokkan menjadi "visual words", lalu gambar direpresentasikan sebagai histogram frekuensi visual words.

### Langkah Kerja
1. Siapkan dataset 3 kategori (misal: kucing, anjing, burung), 30 gambar per kategori.
2. Extract SIFT keypoints dan descriptors dari seluruh gambar.
3. Lakukan k-Means clustering pada semua descriptors (k=50, 100, 200).
4. Buat histogram visual words untuk setiap gambar.
5. Bagi data: 70% training, 30% testing.
6. Train SVM classifier pada histogram.
7. Predict pada test set dan hitung akurasi.
8. Buat confusion matrix.
9. Variasikan jumlah cluster (k) dan plot akurasi vs k.
10. Bandingkan dengan KNN classifier.

### Analisis Percobaan 5
- Berapa nilai k optimal untuk visual vocabulary?
- Bagaimana akurasi SVM vs KNN?
- Pada kategori mana classifier paling sering salah? Mengapa?
- Apa kelemahan BoVW dibandingkan deep learning?

---

## Percobaan 6: OCR — Optical Character Recognition

### Tujuan
Mengimplementasikan OCR menggunakan Tesseract untuk mengekstrak teks dari gambar.

### Dasar Teori
OCR mengubah gambar teks menjadi string yang dapat diproses komputer. Pipeline: preprocessing (binarisasi, denoising) → layout analysis → character recognition → post-processing.

### Langkah Kerja
1. Install Tesseract OCR engine dan pytesseract.
2. Load gambar dokumen (printed text, scan buku, foto KTP).
3. Preprocessing: grayscale → threshold → denoise.
4. Jalankan `pytesseract.image_to_string()`.
5. Coba berbagai preprocessing: Otsu, Adaptive, Gaussian blur.
6. Bandingkan akurasi OCR per preprocessing.
7. Uji OCR pada gambar teks miring → coba deskewing terlebih dahulu.
8. Uji OCR pada gambar resolusi rendah vs tinggi.
9. Uji OCR bahasa Indonesia (`lang='ind'`).
10. Gunakan `image_to_data()` untuk mendapatkan bounding box per kata.

### Analisis Percobaan 6
- Preprocessing mana yang menghasilkan OCR paling akurat?
- Berapa minimum resolusi agar OCR masih akurat?
- Apakah teks miring berhasil dibaca setelah deskewing?
- Bagaimana performa OCR pada teks bahasa Indonesia vs Inggris?

---

## Percobaan 7: Scene Text Detection (EAST)

### Tujuan
Mendeteksi lokasi teks dalam gambar scene (bukan dokumen) menggunakan model EAST.

### Dasar Teori
EAST (Efficient and Accurate Scene Text Detector) menggunakan FCN untuk memprediksi score map dan geometry (rotated box) secara dense, lalu NMS untuk menghasilkan bounding box final.

### Langkah Kerja
1. Download model EAST (`frozen_east_text_detection.pb`).
2. Load model dengan `cv2.dnn.readNet()`.
3. Resize gambar ke kelipatan 32 (misal 320×320).
4. Forward pass → ambil scores dan geometry.
5. Implementasikan NMS untuk menggabungkan deteksi overlapping.
6. Gambar bounding box teks pada gambar asli.
7. Uji pada foto papan nama / rambu jalan.
8. Uji pada foto produk / buku.
9. Kombinasikan EAST + Tesseract: deteksi → crop → OCR.
10. Uji pipeline end-to-end (EAST + OCR) pada 5 gambar berbeda.

### Analisis Percobaan 7
- Pada tipe gambar apa EAST bekerja paling baik?
- Berapa confidence threshold optimal?
- Apakah teks vertikal atau miring terdeteksi?
- Seberapa akurat pipeline EAST + OCR end-to-end?

---

## Percobaan 8: Pedestrian Detection (HOG + SVM)

### Tujuan
Mendeteksi pejalan kaki menggunakan metode klasik Histogram of Oriented Gradients.

### Dasar Teori
HOG menghitung distribusi orientasi gradien dalam cell kecil, kemudian menormalisasi dalam block. Feature HOG sangat diskriminatif untuk bentuk manusia berjalan.

### Langkah Kerja
1. Load gambar yang berisi pejalan kaki.
2. Buat HOG descriptor: `cv2.HOGDescriptor()`.
3. Set SVM detector: `getDefaultPeopleDetector()`.
4. Jalankan `detectMultiScale()` dengan parameter default.
5. Gambar bounding box pada pejalan kaki terdeteksi.
6. Variasikan `winStride` (4,4), (8,8), (16,16) dan amati perbedaan.
7. Variasikan `padding` dan `scale` parameter.
8. Uji pada gambar kerumunan vs satu orang.
9. Uji pada video/webcam (real-time pedestrian detection).
10. Bandingkan HOG + SVM dengan YOLO pedestrian detection.

### Analisis Percobaan 8
- Berapa pejalan kaki terdeteksi dan berapa false positive?
- Bagaimana `winStride` mempengaruhi akurasi vs kecepatan?
- Pada jarak berapa pejalan kaki masih terdeteksi?
- Seberapa besar perbedaan performa HOG vs YOLO?

---

## Percobaan 9: Hand Gesture Recognition (MediaPipe)

### Tujuan
Mendeteksi tangan dan mengenali gesture menggunakan MediaPipe Hands.

### Dasar Teori
MediaPipe Hands mendeteksi 21 landmark 3D per tangan secara real-time. Dari posisi landmark, kita dapat mengklasifikasikan gesture (finger counting, thumbs up, peace sign, dll).

### Langkah Kerja
1. Install MediaPipe: `pip install mediapipe`.
2. Load gambar tangan dan deteksi landmarks.
3. Visualisasikan 21 landmark + connections pada gambar.
4. Implementasikan finger counting: hitung jari yang terangkat berdasarkan posisi landmark.
5. Definisikan 5 gesture berbeda dan buat classifier sederhana (rule-based).
6. Gambar label gesture pada gambar.
7. Uji finger counter pada gambar dengan background bervariasi.
8. Implementasikan real-time gesture recognition dari webcam.
9. Buat simple hand-controlled UI (misal: move cursor berdasarkan posisi tangan).
10. Hitung FPS dan evaluasi performa real-time.

### Analisis Percobaan 9
- Berapa akurasi finger counting pada berbagai posisi tangan?
- Gesture mana yang paling sulit dikenali? Mengapa?
- Berapa FPS rata-rata pada webcam?
- Pada kondisi pencahayaan apa MediaPipe gagal?

---

## Percobaan 10: Evaluation — Metrics dan Benchmark

### Tujuan
Menghitung dan memvisualisasikan metrics evaluasi untuk berbagai recognition tasks.

### Dasar Teori
Evaluasi kuantitatif sangat penting untuk membandingkan model dan memilih pendekatan terbaik. Classification metrics (accuracy, precision, recall, F1), Detection metrics (IoU, mAP), dan Recognition metrics (TAR, FAR, ROC) memberikan perspektif berbeda.

### Langkah Kerja
1. Kumpulkan prediksi dan ground truth dari percobaan sebelumnya.
2. Hitung accuracy, precision, recall, F1-score.
3. Buat dan visualisasikan confusion matrix.
4. Plot precision-recall curve.
5. Plot ROC curve (untuk face recognition: similarity score vs label).
6. Hitung IoU untuk face detection bounding boxes.
7. Implementasikan mAP calculation sederhana.
8. Bandingkan metrik antar metode (tabel ringkasan).
9. Visualisasikan semua metrik dalam satu dashboard (subplots).
10. Tulis analisis tertulis: metode mana yang terbaik dan mengapa.

### Analisis Percobaan 10
- Metode mana yang memiliki F1-score tertinggi?
- Pada threshold berapa ROC curve menunjukkan trade-off terbaik?
- Apakah mAP konsisten dengan pengamatan visual?
- Bagaimana Anda akan memilih metode terbaik untuk deployment?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Perbandingan metode klasik vs deep learning untuk face detection.
2. Perbandingan LBPH vs DeepFace untuk face recognition.
3. Efektivitas BoVW untuk object classification.
4. Performa OCR dan faktor yang mempengaruhinya.
5. Evaluasi metrics dan pemilihan metode terbaik.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`) dengan markdown + kode + output.
- **Naming**: `NIM_Nama_Modul06.ipynb`
