# NotebookLM Prompts — Modul 6: Recognition (Pengenalan)

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 6: Recognition (Pengenalan). Referensi Szeliski (2022) Ch.6. Tiap slide ~500 kata, sertakan diagram, formula, dan kode OpenCV/Python.

**Slide 1** — Judul "Modul 6: Recognition (Pengenalan)", subtitle "Dari Piksel ke Identitas", ilustrasi face recognition pipeline, referensi Szeliski Ch.6.

**Slide 2** — Taxonomi Recognition: Instance Recognition (wajah tertentu, landmark) vs Category Recognition (kelas seperti kucing/mobil). Tingkatan: Classification (label tunggal) → Detection (bbox+label) → Segmentation (per piksel) → Identification (entitas spesifik). Diagram hierarki.

**Slide 3** — Face Detection Klasik: Viola-Jones/Haar Cascade. Feature(x)=Σ_white I(i)−Σ_black I(j). Integral Image untuk komputasi cepat. AdaBoost seleksi fitur diskriminatif. Cascade Classifier: reject cepat di tahap awal. cv2.CascadeClassifier, detectMultiScale(scaleFactor, minNeighbors).

**Slide 4** — Face Detection DNN: SSD face model (Caffe), MTCNN, RetinaFace. cv2.dnn.readNetFromCaffe(), blobFromImage(1.0,(300,300),(104,177,123)), net.forward(). Perbandingan Haar vs DNN: akurasi, kecepatan, robustness terhadap pose/oklusi.

**Slide 5** — Face Recognition Pipeline: Detection → Alignment (landmarks) → Embedding (feature vector) → Matching (jarak Euclidean/cosine). Klasik: Eigenfaces (PCA), Fisherfaces (LDA), LBPH (tekstur lokal). Deep Learning: FaceNet (triplet loss, 128-D), ArcFace (angular margin), DeepFace.

**Slide 6** — Formula: Triplet Loss = max(0, ‖f(a)−f(p)‖²−‖f(a)−f(n)‖²+α). LBPH: cv2.face.LBPHFaceRecognizer_create(), train(faces, labels), predict(test_face). Eigenfaces: PCA projection ke ruang eigenvector wajah. Fisherfaces: LDA maksimalkan separasi kelas.

**Slide 7** — Object Recognition: BoVW (Bag of Visual Words): SIFT keypoints → k-Means clustering → visual vocabulary → histogram frekuensi → SVM/KNN. Deep Learning: ResNet, EfficientNet, ViT. Transfer learning untuk domain spesifik. Spatial Pyramid Matching: grid multi-level + histogram fitur per cell.

**Slide 8** — Scene Recognition: GIST descriptor (representasi holistik), Places CNN (365 kategori). Spatial Pyramid Kernel: K(x,y)=Σ_l (1/2^{L-l}) Σ_i min(H_l^x(i), H_l^y(i)). Indoor vs outdoor challenges. OCR Pipeline: preprocessing (binarize, deskew, denoise) → text detection (EAST) → Tesseract recognition.

**Slide 9** — Percobaan 1–3: Face Detection Haar (scaleFactor 1.05/1.1/1.3, minNeighbors 3/5/7, gambar grup). Face Detection DNN (confidence threshold, perbandingan metode). Face Recognition LBPH (training database min 5 orang, prediction + confidence score).

**Slide 10** — Percobaan 4–6: Face Recognition Eigenfaces (PCA projection, visualisasi eigenface). Face Landmark Detection (68 landmarks, facial feature extraction — dlib/Mediapipe). OCR Preprocessing (binarisasi, denoising, deskewing — perbandingan metode).

**Slide 11** — Percobaan 7–9: Tesseract OCR (multi-language, psm modes, konfigurasi). Scene Text Detection EAST (deteksi lokasi teks → OCR pipeline end-to-end). Pedestrian Detection HOG (cv2.HOGDescriptor, setSVMDetector, detectMultiScale, parameter tuning, sliding window).

**Slide 12** — Percobaan 10–12: Vehicle Detection + counting + klasifikasi. Hand Gesture Recognition MediaPipe (21 landmarks, finger counting, real-time gesture). Object Classification BoVW (SIFT → KMeans vocabulary → histogram → SVM pipeline).

**Slide 13** — Percobaan 13–15: Scene Recognition + Spatial Pyramid Matching. Face Embedding Distance (Euclidean vs Cosine similarity, threshold verification/identification). Classification Metrics (accuracy, precision, recall, F1, confusion matrix, multi-class).

**Slide 14** — Percobaan 16–18: Detection Metrics (IoU, mAP, precision-recall curve). Recognition ROC Curve (ROC, AUC, EER, threshold tuning). Multi-Face Tracking (deteksi per frame + ID assignment + trajectory).

**Slide 15** — Percobaan 19–20: Recognition Pipeline Lengkap (detection→alignment→embedding→matching→evaluation chain). Proyek recognition sistem terintegrasi. Setup: opencv-contrib-python, pytesseract, mediapipe, deepface, scikit-learn; install Tesseract OCR engine.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 6, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: Haar face detection, DNN face detection, LBPH recognition, Eigenfaces, face landmarks, OCR preprocessing, Tesseract, EAST text detection, HOG pedestrian, vehicle detection. Grid thumbnail. Tabel nama percobaan, file, metode kunci.

**Slide 17** — Rekap percobaan 11–20: Gesture MediaPipe, BoVW classification, scene recognition, face embedding, classification metrics, detection metrics, ROC curve, multi-face tracking, full pipeline, proyek. Tabel library utama, konsep kunci, output.

**Slide 18** — Analisis mendalam: Haar vs DNN — mengapa DNN lebih robust terhadap oklusi? Pipeline recognition berbasis embedding vs berbasis template. Kapan BoVW masih relevan versus deep features? LBPH vs Eigenfaces: interpretasi, kecepatan, memori.

**Slide 19** — Koneksi antar modul: Face detection (Modul 6) menggunakan fitur DNN dari Modul 5. BoVW menggunakan SIFT dari Modul 7. Tracking recognition dari Modul 9 (optical flow + ID). OCR bergantung preprocessing Modul 3. Pipeline recognition = fondasi smart surveillance.

**Slide 20** — Best practices: Normalisasi wajah sebelum recognition (align ke landmark). Threshold embedding: tidak terlalu ketat (banyak false reject) atau longgar (banyak false accept). ROC dan EER untuk memilih threshold optimal. Augmentasi untuk dataset wajah kecil.

**Slide 21** — Aplikasi nyata: Absensi wajah kampus (face detection + recognition + logging). Smart parking (OCR plat nomor). Penjaga toko (pedestrian counting). Sign language interpreter (gesture MediaPipe). BoVW untuk museum artefak recognition. End-to-end pipeline masing-masing.

**Slide 22** — Perbandingan recognition methods: tabel LBPH/Eigenfaces/FaceNet — training time, inference time, memori, akurasi LFW. Kapan LBPH cukup (embedded, offline), kapan FaceNet/ArcFace diperlukan (server, akurasi tinggi).

**Slide 23** — Checklist kompetensi: face detection (Haar+DNN), recognition (LBPH+Eigenfaces), landmarks, OCR pipeline, pedestrian detection, gesture MediaPipe, BoVW, metrics (accuracy/mAP/ROC/AUC/EER). Self-assessment tabel per percobaan.

**Slide 24** — Kuis: (1) Nilai confidence LBPH: makin kecil makin (mirip/berbeda)? (2) Komponen integral image digunakan untuk? (3) Apa yang dikalkulasi EER pada ROC curve? (4) Dimensi embedding FaceNet? (5) EAST detector output berupa?

**Slide 25** — Diskusi: Kapan BoVW cocok vs deep features untuk retrieval? Mengapa angle loss pada ArcFace lebih baik dari Euclidean FaceNet? Bagaimana mengatasi illumination variation pada face recognition? Tradeoff Tesseract vs EasyOCR untuk bahasa Indonesia.

**Slide 26** — Project "Sistem Recognition Terpadu". 10 soal cerita: Absensi Wajah Kampus (face detection+recognition+logging), Pembaca KTP (OCR+parsing JSON), Keamanan Smart Home (face+pedestrian+emosi), Penerjemah Teks Real-time (EAST+OCR+translate), Counting Pengunjung Mall (HOG+YOLO+crossing line), Pengenalan Isyarat (MediaPipe+klasifikasi gestur), Smart Parking Gate (OCR plat nomor+gate logic), Museum Guide (BoVW+DL hybrid), Monitoring Gudang (pedestrian+helmet+face), Gesture Game Controller (5+ gestur → aksi game). Deliverable: .py, output/, laporan.

**Slide 27** — 15 improvisasi: Multi-face Attendance System, Emotion-aware Recognition, Age-Gender Profiling, Anti-spoofing, Multi-language OCR, Receipt/Invoice Parser, License Plate Recognition, Sign Language Translator, Gesture-controlled Presentation, Product Recognition, Document Scanner+OCR PDF, Face Clustering (tanpa label), Pedestrian Counter, Custom Classifier Flask UI, Interactive Recognition Dashboard.

**Slide 28** — Rubrik Project: Fungsionalitas 35%, Integrasi Percobaan 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Format: ZIP NIM_Nama_Project06.zip. Deadline 1 minggu. Bonus +5 demo real-time webcam, +3 sistem pipeline lengkap dengan evaluasi.

**Slide 29** — Tugas Video: 10–20 menit, screen+face-cam. Pembukaan (2 mnt): NIM/nama. Materi (10–15 mnt): taxonomi, Viola-Jones vs DNN, LBPH/Eigenfaces/FaceNet, BoVW, OCR, HOG, metrics — wajib diagram per topik. Demo 20 Percobaan LIVE (60–80 mnt) — 2/percobaan termasuk real-time face dan gesture. Demo Project (10–15 mnt). Analisis & Penutup (5 mnt).

**Slide 30** — Rubrik Video: Pembukaan (5), Materi (10), 20 Percobaan (40 — 2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 demo real-time recognition lengkap, +3 multi-method comparison. Penalti −2/percobaan tidak tampil. "Kenali Wajah, Baca Teks, Pahami Gestur — Bangun Sistem Cerdas!"
