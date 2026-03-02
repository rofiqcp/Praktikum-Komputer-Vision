# NotebookLM Prompts — Modul 10: Recognition (Pengenalan)

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet Bagian 1)

Buat 15 slide presentasi akademik Modul 10: Recognition (Pengenalan). Referensi Szeliski (2022) Ch.6, Mastering OpenCV 4. Tiap slide ~500 kata, sertakan diagram dan kode Python.

**Slide 1** — Judul "Modul 10: Recognition (Pengenalan)", subtitle "Dari Deteksi ke Identifikasi: Face, Object, Text Recognition", ilustrasi pipeline detection→recognition→identification, referensi Szeliski Ch.6.

**Slide 2** — Taxonomi Recognition: Classification (kategori gambar), Detection (lokalisasi bbox), Segmentation (per-piksel), Identification (identitas spesifik). Instance vs Category recognition. Diagram hierarki recognition tasks.

**Slide 3** — Face Detection Haar Cascade: Fitur Haar-like (edge, line, center-surround), integral image O(1), AdaBoost feature selection, cascade classifier (multi-stage reject). cv2.CascadeClassifier + detectMultiScale parameters (scaleFactor, minNeighbors).

**Slide 4** — DNN Face Detection: OpenCV DNN module readNetFromCaffe, SSD-based model, confidence threshold. Perbandingan Haar vs DNN: akurasi (95% vs 99%), kecepatan, false positive rate. Tabel waktu inferensi.

**Slide 5** — Face Recognition Pipeline: Detect → Align → Extract → Match. Eigenfaces (PCA): x≈x̄+Σwᵢuᵢ, variasi utama wajah. Fisherfaces (LDA): maksimalkan between-class variance. LBPH: LBP(xc,yc)=Σs(gp-gc)·2^p, histogram per region. Tabel perbandingan 3 metode.

**Slide 6** — Face Embedding modern: FaceNet (128-d, triplet loss), ArcFace (512-d, angular margin). Jarak Euclidean d(a,b)=||a-b||₂ dan Cosine similarity sim=a·b/(|a||b|). Threshold-based matching. Diagram embedding space: wajah sama dekat, wajah beda jauh.

**Slide 7** — Face Landmark: 68-point model (dlib). Region: jawline (0-16), alis (17-26), hidung (27-35), mata (36-47), mulut (48-67). Aplikasi: alignment, expression analysis, face morphing, gaze estimation. Percobaan 5.

**Slide 8** — OCR Pipeline: Input → Preprocessing → Detection → Segmentation → Recognition → Output. Preprocessing: binarisasi (Otsu, Adaptive), denoising (Median, NLM), deskew (moment-based rotation). Tesseract OCR: LSTM-based, 100+ bahasa. Percobaan 6-7.

**Slide 9** — Scene Text Detection: MSER (Maximally Stable Extremal Regions) — region intensitas stabil. Morphology-based: Sobel gradient → threshold → horizontal dilasi → contour filter. EAST text detector (DNN). Percobaan 8.

**Slide 10** — HOG Pedestrian Detection: (1) gradient magnitude dan orientasi, (2) 8×8 cell histogram 9 bins, (3) 2×2 block L2-norm, (4) gabung→SVM. cv2.HOGDescriptor + getDefaultPeopleDetector(). Kelebihan: invariant pencahayaan global. Percobaan 9.

**Slide 11** — Object Classification BoVW: Ekstrak fitur lokal (SIFT/ORB) → K-Means clustering → visual dictionary → histogram per gambar → SVM/KNN classifier. Analog Bag of Words di NLP. Scene recognition: color histogram + LBP texture + GIST. Percobaan 12-13.

**Slide 12** — Hand Gesture Recognition: Skin color segmentation (HSV H:0-20, S:30-170), convex hull, convexity defects (celah antar jari). Hitung jari: sudut defect < π/2 dan depth > threshold. Percobaan 11.

**Slide 13** — Percobaan 1-4: Haar cascade face detection (variasi scaleFactor/minNeighbors), DNN face detection konsep (fallback + timing), LBPH face recognition (LBP manual + train recognizer), Eigenfaces (PCA/SVD + rekonstruksi + cv2.face.EigenFaceRecognizer).

**Slide 14** — Percobaan 5-10: Face landmarks (estimasi + 68-point diagram), OCR preprocessing (Otsu/Adaptive/denoise/deskew), Tesseract OCR (pipeline + projection profile), Scene text (MSER + morphology), Pedestrian HOG (gradient + detectMultiScale), Vehicle detection (contour + Haar).

**Slide 15** — Percobaan 11-14: Gesture (HSV segmentasi + convexity defects + finger count), BoVW (ORB + K-Means + histogram klasifikasi), Scene recognition (color + LBP + distance), Face embedding (embedding sederhana + distance matrix heatmap).

---

## PROMPT 2 — Slide 16–30 (Materi Lanjutan + Analisis)

Lanjutkan slide presentasi Modul 10: Recognition (Pengenalan), Slide 16–30. Tiap slide ~500 kata. Fokus analisis mendalam, metrik evaluasi, advanced topics, dan rekap.

**Slide 16** — Classification Metrics: Confusion Matrix (TP/TN/FP/FN). Accuracy=(TP+TN)/total, Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2PR/(P+R). Accuracy menyesatkan pada imbalanced data. Macro vs micro averaging. Percobaan 15.

**Slide 17** — Detection Metrics: IoU=|A∩B|/|A∪B|. Threshold standar: IoU≥0.5 (mAP@50), IoU≥0.75 (mAP@75). Average Precision = area under PR curve. Mean AP = rata-rata AP per kelas. NMS: suppress overlapping bbox. Percobaan 16.

**Slide 18** — ROC Curve dan AUC: TPR vs FPR pada berbagai threshold. AUC=1.0 ideal, 0.5=random. Score distribution: genuine (same person) vs impostor (different person). EER (Equal Error Rate): titik FPR=FNR. Percobaan 17.

**Slide 19** — Multi-face Tracking: Detection per frame + ID assignment via center distance matching. Tantangan: oklusi, new entry/exit, ID switch. Hungarian algorithm untuk optimal assignment. Kalman filter untuk prediksi posisi. Percobaan 18.

**Slide 20** — Recognition Pipeline End-to-End: (1) Detect: Haar/DNN lokalisasi wajah. (2) Align: crop + resize + equalizeHist. (3) Embed: extract feature vector. (4) Match: Euclidean/Cosine vs database + threshold. (5) Result: identity + confidence. Percobaan 19.

**Slide 21** — Rekap Percobaan 1-10: (1) Haar face detection + parameter variasi, (2) DNN konsep + Haar fallback, (3) LBPH recognition + training, (4) Eigenfaces PCA + rekonstruksi, (5) Face landmarks 68-point, (6) OCR preprocessing, (7) Tesseract OCR, (8) Scene text MSER, (9) HOG pedestrian, (10) Vehicle detection.

**Slide 22** — Rekap Percobaan 11-20: (11) Gesture finger count, (12) BoVW classification, (13) Scene recognition, (14) Face embedding distance, (15) Classification metrics CM/P/R/F1, (16) Detection metrics IoU/mAP, (17) ROC curve AUC, (18) Multi-face tracking, (19) Pipeline lengkap, (20) Proyek sistem recognition.

**Slide 23** — Analisis mendalam: Haar 24fps vs DNN 10fps tapi DNN lebih akurat. LBPH robust terhadap lighting vs Eigenfaces butuh uniform lighting. Embedding modern (ArcFace) 99.8% LFW tapi butuh GPU. BoVW vs CNN: trade-off akurasi vs simplicity. HOG+SVM vs YOLO: classical vs deep.

**Slide 24** — Perbandingan Face Recognition: LBPH (gallery size<1000, no GPU), Eigenfaces (linear, fast, variant lighting), Fisherfaces (better class separation), ArcFace (large-scale, GPU). Tabel: metode, akurasi LFW, kecepatan, memory, kebutuhan data.

**Slide 25** — OCR Best Practices: DPI optimal 300, binarisasi adaptive untuk pencahayaan tidak rata, NLM denoising untuk noise tinggi, deskew untuk rotated text. Language model + post-processing (spell check). Tesseract vs EasyOCR vs PaddleOCR comparison.

**Slide 26** — Evaluasi yang Tepat: Classification → accuracy, F1, confusion matrix. Detection → mAP@50, IoU analysis. Recognition/Verification → ROC-AUC, EER. Retrieval → mAP, recall@K. Jangan gunakan accuracy saja untuk imbalanced data.

**Slide 27** — Advanced Recognition Concepts: One-shot learning (Siamese network), Zero-shot learning (CLIP), Open-set recognition (reject unknown), Continual learning (add new classes tanpa re-train). Cross-domain recognition challenges.

**Slide 28** — Real-world Deploy Challenges: Lighting variation, pose variation, partial occlusion, aging, disguise, scale variation. Anti-spoofing: liveness detection (blink, head turn, depth analysis). Privacy concerns: face recognition ethics.

**Slide 29** — Integrasi dengan Modul Lain: Modul 3 preprocessing → OCR quality. Modul 4 feature detection → BoVW. Modul 9 DNN → face detection, embedding. Modul 11 depth → 3D face recognition. Modul 12 reconstruction → face model.

**Slide 30** — Rangkuman: Recognition = klasifikasi, deteksi, identifikasi. Pipeline: detect → align → embed → match. Evaluasi: metrics yang tepat sesuai task. Trend: deep learning mendominasi tapi classical methods masih relevan untuk edge deploy.

---

## PROMPT 3 — Slide 31–45 (Advanced + Project + Video)

Lanjutkan slide presentasi Modul 10: Recognition (Pengenalan), Slide 31–45. Tiap slide ~500 kata. Fokus advanced applications, project guidance, dan tugas video.

**Slide 31** — Advanced Face Detection: Multi-scale detection pyramid. Anchor-based vs anchor-free detectors. RetinaFace: joint face detection + landmark + 3D face. TinaFace: large-scale dalam kerumunan. Speed-accuracy tradeoff diagram.

**Slide 32** — Advanced Face Recognition: Large-scale FR (millions of identities). Hard example mining, curriculum learning. Quality-aware recognition: blur/pose/illumination scoring. Tabel: state-of-art methods benchmark pada MegaFace, IJB-C.

**Slide 33** — OCR Advanced: CRNN (CNN+RNN) untuk sequence recognition. Attention-based OCR (Transformers). Multilingual OCR challenges. Handwriting recognition vs printed text. Scene text: irregular text (curved, rotated).

**Slide 34** — Video Understanding: Action recognition (C3D, I3D, SlowFast). Re-identification across cameras. Long-term tracking (DeepSORT). Activity detection vs recognition. Temporal modeling importance.

**Slide 35** — Project Guidance Soal 1-3: (1) Sistem Keamanan: face DB + anti-spoofing + logging + alert. (2) Parkir ANPR: detect plate → preprocess → OCR → DB lookup. (3) Retail Analytics: pedestrian count + demographics + heatmap.

**Slide 36** — Project Guidance Soal 4-6: (4) Anti-Cheating: face verify + head pose + multi-face alert. (5) Perpustakaan Digital: document scan → OCR → compile. (6) Traffic Monitor: vehicle + pedestrian count + speed estimation.

**Slide 37** — Project Guidance Soal 7-10: (7) Product Recognition: BoVW/CNN classify + barcode scan. (8) Facial Health: landmark + symmetry analysis. (9) Meeting Attendance: multi-face + recognition + report. (10) Gesture Presentation: real-time gesture → action mapping.

**Slide 38** — Best Practices Project: (a) Start simple, iterate. (b) Modular code (detect → recognize → evaluate). (c) Handle edge cases (no face, multiple faces, unknown). (d) Measure performance (FPS, accuracy, memory). (e) Document assumptions and limitations.

**Slide 39** — Rubrik Project detail: Fungsionalitas (35%) — semua fitur bekerja, handle error, output benar. Integrasi Percobaan (20%) — gunakan min 5 teknik dari 20 percobaan. Kualitas Kode (15%) — def/function style, comments, clean. Dokumentasi (15%) — README, docstring. Kreativitas (15%) — improvisasi, UI, extra features.

**Slide 40** — Common Mistakes: (1) Tidak handle gambar None. (2) Threshold terlalu ketat/longgar. (3) Tidak normalize embedding. (4) Evaluasi pada training data. (5) Tidak save output. Tips debugging: print intermediate results, visualize each step.

**Slide 41** — Tugas Video Overview: 15-25 menit, MP4 720p+. Struktur: Pembukaan (1-2 min) → Materi min 5 topik (5-8 min) → Demo min 5 percobaan (5-8 min) → Project demo (3-5 min) → Penutup (1-2 min).

**Slide 42** — Demo Percobaan Tips: Jalankan program, tunjukkan output, jelaskan output. Highlight: face detection params comparison, LBPH vs Eigenfaces, OCR preprocessing difference, HOG gradient, BoVW histogram per class, ROC curve interpretation.

**Slide 43** — Video Content Checklist: ☐ Perkenalan jelas ☐ Min 5 topik materi explained ☐ Min 5 percobaan dijalankan ☐ Output folder ditunjukkan ☐ Project demo complete ☐ Analisis hasil ☐ Kesimpulan.

**Slide 44** — Koneksi Industry: Face recognition: smartphone unlock, airport gates, payment. OCR: document scanning, receipt processing, license plate. HOG: surveillance, ADAS (Advanced Driver Assistance). Scene recognition: photo organization, self-driving context.

**Slide 45** — Penutup Modul 10: Recognition = core task CV. Pipeline standard untuk semua task: detect → process → classify/identify. Evaluasi proper = kunci deployment. Classical (Haar, HOG, LBP) masih relevan, deep learning (ArcFace, YOLO, CRNN) = state of art. Next: Modul 11 Structure from Motion & Depth.
