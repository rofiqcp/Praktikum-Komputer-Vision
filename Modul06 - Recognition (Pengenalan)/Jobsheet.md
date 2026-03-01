# JOBSHEET MODUL 6: RECOGNITION (PENGENALAN)

---

## Tujuan Praktikum
1. Memahami dan mengimplementasikan face detection (Haar Cascade dan DNN).
2. Mengimplementasikan face recognition (LBPH, Eigenfaces).
3. Memahami face landmark detection dan aplikasinya.
4. Melakukan OCR preprocessing dan recognition (Tesseract).
5. Mengimplementasikan scene text detection.
6. Mengimplementasikan pedestrian dan vehicle detection.
7. Mengimplementasikan hand gesture recognition menggunakan MediaPipe.
8. Mengimplementasikan object classification dengan Bag of Visual Words (BoVW).
9. Memahami scene recognition dan spatial pyramid matching.
10. Memahami face embedding dan distance metrics untuk verifikasi/identifikasi.
11. Menghitung dan menganalisis classification metrics (accuracy, precision, recall, F1).
12. Menghitung dan menganalisis detection metrics (IoU, mAP).
13. Menganalisis ROC curve dan AUC untuk recognition systems.
14. Mengimplementasikan multi-face tracking.
15. Membangun recognition pipeline lengkap (detection + recognition + evaluation).
16. Mengembangkan proyek recognition sistem terintegrasi.

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

## Percobaan 11: Hand Gesture Recognition (MediaPipe)

### Tujuan
Mengenali gestur tangan menggunakan MediaPipe hand landmarks dan mengimplementasikan finger counting.

### Dasar Teori
MediaPipe Hands mendeteksi 21 landmark 3D per tangan secara real-time. Dengan menganalisis posisi relatif landmark (tip vs PIP joint), kita dapat menentukan jari mana yang terangkat dan mengklasifikasikan gestur tangan.

### Langkah Kerja
1. Install dan import MediaPipe Hands.
2. Load gambar tangan dan deteksi 21 landmarks.
3. Visualisasikan landmarks dan connections pada gambar.
4. Implementasikan logika finger counting berdasarkan posisi tip vs PIP.
5. Uji finger counting untuk angka 0-5 pada satu tangan.
6. Definisikan 5 gestur berbeda (fist, open palm, thumbs up, peace, pointing).
7. Buat classifier rule-based berdasarkan status jari.
8. Uji gesture recognition pada gambar dengan background bervariasi.
9. Hitung akurasi gesture recognition pada 20 gambar uji.
10. Tampilkan hasil deteksi dengan label gesture dan confidence.

### Analisis Percobaan 11
- Berapa akurasi finger counting pada berbagai posisi tangan?
- Gesture mana yang paling sulit dikenali? Mengapa?
- Bagaimana background mempengaruhi akurasi deteksi?
- Apa keterbatasan pendekatan rule-based untuk gesture recognition?

---

## Percobaan 12: Object Classification — Bag of Visual Words (BoVW)

### Tujuan
Mengimplementasikan klasifikasi objek menggunakan pipeline Bag of Visual Words: SIFT → KMeans → Histogram → SVM.

### Dasar Teori
BoVW menganalogikan gambar dengan dokumen teks. Keypoint descriptor (SIFT) dikelompokkan menjadi "visual words" menggunakan KMeans clustering. Setiap gambar direpresentasikan sebagai histogram frekuensi visual words, lalu diklasifikasi dengan SVM.

### Langkah Kerja
1. Siapkan dataset minimal 3 kategori objek, 20+ gambar per kategori.
2. Ekstrak SIFT keypoints dan descriptors dari seluruh gambar.
3. Gabungkan semua descriptors dan lakukan KMeans clustering (k=50).
4. Bangun histogram visual words untuk setiap gambar.
5. Bagi dataset: 70% training, 30% testing.
6. Train SVM classifier pada histogram training.
7. Predict test set dan hitung akurasi.
8. Buat dan visualisasikan confusion matrix.
9. Variasikan jumlah cluster (k=25, 50, 100) dan bandingkan akurasi.
10. Bandingkan hasil SVM dengan KNN classifier.

### Analisis Percobaan 12
- Berapa nilai k optimal untuk visual vocabulary?
- Bagaimana akurasi SVM dibandingkan KNN?
- Pada kategori mana classifier paling sering salah? Mengapa?
- Apa kelemahan utama pendekatan BoVW?

---

## Percobaan 13: Scene Recognition

### Tujuan
Mengimplementasikan scene recognition menggunakan konsep spatial pyramid matching.

### Dasar Teori
Scene recognition mengklasifikasikan keseluruhan scene (indoor, outdoor, pantai, hutan, dll). Spatial Pyramid Matching membagi gambar menjadi grid bertingkat (1×1, 2×2, 4×4) dan menghitung histogram fitur per cell, menghasilkan representasi yang mempertahankan informasi spasial.

### Langkah Kerja
1. Siapkan dataset scene minimal 4 kategori (indoor, outdoor, pantai, kota).
2. Ekstrak fitur global (color histogram, texture) dari setiap gambar.
3. Implementasikan spatial pyramid: bagi gambar menjadi grid 1×1, 2×2, 4×4.
4. Hitung histogram fitur per cell di setiap level.
5. Gabungkan histogram dengan bobot per level (weighted concatenation).
6. Train SVM classifier pada fitur spatial pyramid.
7. Evaluasi pada test set dan hitung akurasi.
8. Bandingkan akurasi spatial pyramid vs histogram global (tanpa grid).
9. Visualisasikan contoh prediksi benar dan salah per kategori.
10. Buat confusion matrix dan analisis pola kesalahan.

### Analisis Percobaan 13
- Berapa peningkatan akurasi spatial pyramid dibandingkan histogram global?
- Kategori scene mana yang paling mudah/sulit dikenali?
- Level pyramid mana yang paling berkontribusi?
- Apa keterbatasan pendekatan ini untuk scene yang ambigu?

---

## Percobaan 14: Face Embedding Distance

### Tujuan
Menghitung jarak antar face embedding untuk verifikasi dan identifikasi wajah.

### Dasar Teori
Face embedding memetakan wajah ke vektor dalam ruang berdimensi tinggi. Jarak antar embedding (Euclidean, Cosine) menentukan kemiripan wajah. Threshold jarak digunakan untuk verifikasi (apakah dua wajah orang yang sama?) dan identifikasi (siapa pemilik wajah?).

### Langkah Kerja
1. Siapkan dataset wajah minimal 5 orang, 5 foto per orang.
2. Ekstrak face embedding (128-D atau 512-D) dari setiap gambar wajah.
3. Hitung jarak Euclidean antar embedding intra-class (orang sama).
4. Hitung jarak Euclidean antar embedding inter-class (orang berbeda).
5. Hitung jarak Cosine untuk perbandingan.
6. Visualisasikan distribusi jarak intra-class vs inter-class.
7. Tentukan threshold optimal untuk verifikasi dari distribusi.
8. Implementasikan face verification: input 2 wajah → same/different.
9. Implementasikan face identification: input wajah → identitas terdekat.
10. Evaluasi akurasi verifikasi dan identifikasi pada test set.

### Analisis Percobaan 14
- Berapa threshold jarak optimal untuk verifikasi?
- Seberapa terpisah distribusi intra-class vs inter-class?
- Metrik jarak mana (Euclidean vs Cosine) yang lebih diskriminatif?
- Pada kondisi apa verifikasi gagal (false accept/reject)?

---

## Percobaan 15: Evaluasi — Classification Metrics

### Tujuan
Menghitung dan memvisualisasikan classification metrics secara detail: accuracy, precision, recall, F1, dan confusion matrix.

### Dasar Teori
Classification metrics mengukur performa model klasifikasi. Accuracy mengukur proporsi prediksi benar. Precision mengukur ketepatan prediksi positif. Recall mengukur kemampuan menangkap seluruh positif. F1-score adalah harmonic mean precision dan recall. Confusion matrix menunjukkan distribusi prediksi per kelas.

### Langkah Kerja
1. Siapkan data prediksi dan ground truth dari percobaan sebelumnya.
2. Hitung accuracy secara manual dan dengan scikit-learn.
3. Hitung precision per kelas (macro, micro, weighted).
4. Hitung recall per kelas (macro, micro, weighted).
5. Hitung F1-score per kelas dan rata-rata.
6. Buat confusion matrix dan visualisasikan dengan heatmap.
7. Analisis kelas dengan precision tinggi tapi recall rendah (dan sebaliknya).
8. Implementasikan classification report lengkap.
9. Bandingkan metrik dari 2+ model/metode berbeda dalam tabel.
10. Visualisasikan perbandingan metrik antar model dalam bar chart.

### Analisis Percobaan 15
- Model/metode mana yang memiliki F1-score tertinggi?
- Apakah ada kelas yang precision-nya sangat berbeda dari recall-nya?
- Pada kasus apa macro vs weighted average berbeda signifikan?
- Metrik mana yang paling relevan untuk use case recognition?

---

## Percobaan 16: Evaluasi — Detection Metrics

### Tujuan
Menghitung dan memvisualisasikan detection metrics: IoU, precision-recall curve, dan mAP.

### Dasar Teori
Detection metrics mengukur performa model deteksi objek. IoU (Intersection over Union) mengukur overlap antara predicted dan ground truth bounding box. Precision-Recall curve menunjukkan trade-off deteksi. mAP (mean Average Precision) adalah metrik standar untuk evaluasi object detection.

### Langkah Kerja
1. Siapkan data bounding box prediksi dan ground truth dari percobaan deteksi.
2. Implementasikan fungsi IoU untuk dua bounding box.
3. Hitung IoU untuk setiap pasangan prediksi-ground truth.
4. Klasifikasikan deteksi sebagai TP/FP berdasarkan IoU threshold (0.5).
5. Hitung precision dan recall pada berbagai confidence threshold.
6. Plot precision-recall curve.
7. Hitung AP (Average Precision) dari PR curve.
8. Ulangi untuk IoU threshold berbeda (0.5, 0.75).
9. Hitung mAP sebagai rata-rata AP per kelas.
10. Bandingkan mAP antar metode deteksi (Haar vs DNN, HOG vs YOLO).

### Analisis Percobaan 16
- Berapa mAP@0.5 dan mAP@0.75 untuk setiap metode?
- Pada confidence threshold berapa trade-off precision-recall optimal?
- Metode deteksi mana yang memiliki IoU rata-rata tertinggi?
- Bagaimana perbedaan performa pada IoU ketat (0.75) vs longgar (0.5)?

---

## Percobaan 17: Recognition ROC Curve

### Tujuan
Menganalisis ROC curve, menghitung AUC, dan melakukan threshold tuning untuk recognition systems.

### Dasar Teori
ROC (Receiver Operating Characteristic) curve menunjukkan trade-off antara True Positive Rate (sensitivity) dan False Positive Rate (1-specificity). AUC (Area Under Curve) mengukur kemampuan diskriminatif model secara keseluruhan. Pemilihan threshold optimal bergantung pada trade-off TPR vs FPR sesuai kebutuhan aplikasi.

### Langkah Kerja
1. Kumpulkan similarity score dan label (match/non-match) dari face recognition.
2. Hitung TPR dan FPR pada berbagai threshold.
3. Plot ROC curve.
4. Hitung AUC (Area Under Curve).
5. Tentukan EER (Equal Error Rate) — titik di mana FAR = FRR.
6. Plot FAR vs FRR curve dan tandai titik EER.
7. Analisis threshold optimal untuk skenario high-security (minimize FAR).
8. Analisis threshold optimal untuk skenario convenience (maximize TPR).
9. Bandingkan ROC curve dari 2+ metode recognition.
10. Buat tabel ringkasan: AUC, EER, threshold optimal per metode.

### Analisis Percobaan 17
- Metode mana yang memiliki AUC tertinggi?
- Berapa EER masing-masing metode?
- Bagaimana threshold berbeda untuk skenario keamanan vs kenyamanan?
- Apakah AUC konsisten dengan pengamatan kualitatif?

---

## Percobaan 18: Multi-Face Tracking

### Tujuan
Mengimplementasikan multi-face tracking: deteksi wajah dan pelacakan antar frame pada video.

### Dasar Teori
Multi-face tracking menggabungkan face detection dengan object tracking untuk melacak identitas wajah secara konsisten antar frame video. Tracker (KCF, CSRT, atau centroid-based) mempertahankan identitas objek antara deteksi, mengurangi kebutuhan deteksi per frame.

### Langkah Kerja
1. Siapkan video atau webcam feed dengan beberapa wajah bergerak.
2. Jalankan face detection pada frame pertama.
3. Inisialisasi tracker (cv2.TrackerCSRT atau centroid tracker) untuk setiap wajah.
4. Pada frame berikutnya, update semua tracker.
5. Re-detect wajah setiap N frame untuk menangani wajah baru/hilang.
6. Assign ID unik untuk setiap wajah yang dilacak.
7. Tampilkan bounding box + ID pada setiap frame.
8. Handle kasus wajah masuk/keluar frame (register/deregister).
9. Hitung dan tampilkan jumlah wajah aktif per frame.
10. Evaluasi konsistensi tracking (apakah ID stabil untuk wajah yang sama?).

### Analisis Percobaan 18
- Seberapa stabil ID tracking untuk wajah yang sama?
- Pada kondisi apa tracker kehilangan target?
- Berapa interval re-detection optimal (N frame)?
- Bagaimana performa tracking saat wajah saling overlap?

---

## Percobaan 19: Recognition Pipeline Lengkap

### Tujuan
Membangun pipeline recognition lengkap yang mengintegrasikan detection, recognition, dan evaluation.

### Dasar Teori
Pipeline recognition lengkap menggabungkan tahap detection (menemukan objek), recognition (mengidentifikasi objek), dan evaluation (mengukur performa). Pipeline yang terintegrasi memungkinkan end-to-end processing dari input gambar/video hingga output hasil recognition dengan metrik evaluasi.

### Langkah Kerja
1. Definisikan pipeline: input → detection → preprocessing → recognition → output.
2. Implementasikan modul detection (face detection DNN).
3. Implementasikan modul preprocessing (alignment, resize, normalization).
4. Implementasikan modul recognition (face recognition LBPH/embedding).
5. Implementasikan modul evaluation (classification metrics).
6. Integrasikan semua modul dalam satu pipeline function.
7. Jalankan pipeline pada dataset test.
8. Hitung metrik end-to-end (akurasi keseluruhan pipeline).
9. Identifikasi bottleneck: modul mana yang paling mempengaruhi error?
10. Visualisasikan hasil pipeline step-by-step untuk beberapa contoh.

### Analisis Percobaan 19
- Berapa akurasi end-to-end pipeline?
- Modul mana yang menjadi bottleneck utama?
- Bagaimana error di detection stage mempengaruhi recognition?
- Apa yang bisa ditingkatkan untuk memperbaiki performa pipeline?

---

## Percobaan 20: Proyek Recognition Sistem

### Tujuan
Membangun sistem recognition lengkap sebagai proyek akhir yang mengintegrasikan seluruh konsep dari modul ini.

### Dasar Teori
Proyek akhir mengintegrasikan seluruh komponen: face detection, face recognition, object classification, text recognition, gesture recognition, tracking, dan evaluation metrics. Sistem harus dirancang modular, teruji, dan terdokumentasi dengan baik.

### Langkah Kerja
1. Tentukan domain aplikasi (absensi, keamanan, OCR, dll).
2. Rancang arsitektur sistem: input → processing → output.
3. Implementasikan modul detection sesuai domain.
4. Implementasikan modul recognition/classification sesuai domain.
5. Implementasikan database/storage untuk data reference.
6. Implementasikan user interface sederhana (OpenCV GUI atau command-line).
7. Integrasikan semua modul menjadi sistem utuh.
8. Uji sistem pada skenario realistis.
9. Hitung metrik evaluasi lengkap (accuracy, precision, recall, F1, atau mAP).
10. Dokumentasikan arsitektur, hasil, dan analisis dalam laporan.

### Analisis Percobaan 20
- Apakah sistem berjalan end-to-end sesuai desain?
- Berapa akurasi/metrik keseluruhan sistem?
- Apa kekuatan dan kelemahan sistem yang dibangun?
- Bagaimana sistem dapat dikembangkan lebih lanjut?

---

## Kesimpulan
Tuliskan kesimpulan berdasarkan:
1. Perbandingan metode klasik vs deep learning untuk face detection.
2. Perbandingan LBPH vs Eigenfaces untuk face recognition.
3. Efektivitas face landmark detection dan aplikasinya.
4. Performa OCR dan faktor yang mempengaruhinya (preprocessing, resolusi, bahasa).
5. Efektivitas scene text detection end-to-end.
6. Perbandingan pedestrian detection (HOG vs DNN) dan vehicle detection.
7. Performa hand gesture recognition dengan MediaPipe.
8. Efektivitas BoVW untuk object classification.
9. Kemampuan scene recognition dengan spatial pyramid matching.
10. Analisis face embedding distance untuk verifikasi dan identifikasi.
11. Evaluasi classification metrics dan perbandingan antar model.
12. Evaluasi detection metrics (IoU, mAP) dan implikasi threshold.
13. Analisis ROC/AUC dan pemilihan threshold optimal.
14. Multi-face tracking dan konsistensi pelacakan.
15. Efektivitas pipeline recognition terintegrasi.
16. Refleksi keseluruhan: pemilihan metode terbaik untuk berbagai skenario recognition.

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah praktikum.
- **Format**: Jupyter Notebook (`.ipynb`) dengan markdown + kode + output.
- **Naming**: `NIM_Nama_Modul06.ipynb`
