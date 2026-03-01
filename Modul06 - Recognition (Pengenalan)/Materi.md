# MATERI MODUL 6: RECOGNITION (PENGENALAN)

---

## 1. Pendahuluan

Recognition (pengenalan) merupakan kemampuan sistem komputer untuk mengenali dan mengkategorikan objek, wajah, teks, scene, dan entitas lainnya dalam citra. Ini adalah salah satu area utama computer vision yang memiliki dampak langsung pada berbagai aplikasi dunia nyata — mulai dari pengenalan wajah di smartphone hingga kendaraan otonom.

**Referensi utama**: Szeliski, *Computer Vision: Algorithms and Applications*, 2nd Edition, **Chapter 6 — Recognition**.

---

## 2. Taxonomi Recognition

### 2.1 Instance Recognition vs Category Recognition
- **Instance Recognition**: Mengenali objek spesifik (misal: wajah seseorang tertentu, landmark tertentu).
- **Category Recognition**: Mengenali kategori objek (misal: "kucing", "mobil", "gedung").

### 2.2 Tingkatan Recognition
1. **Classification** — Apa yang ada di gambar? (label tunggal)
2. **Detection** — Di mana objek berada? (bounding box + label)
3. **Segmentation** — Piksel mana milik objek mana?
4. **Identification** — Siapa/apa entitas spesifik?

---

## 3. Face Detection

### 3.1 Viola-Jones (Haar Cascade)
Teknik klasik menggunakan:
- **Haar-like Features**: Pola intensitas sederhana (edge, line, center-surround).
- **Integral Image**: Komputasi cepat area features.
- **AdaBoost**: Pemilihan fitur diskriminatif.
- **Cascade Classifier**: Rangkaian classifier bertingkat — reject cepat pada tahap awal.

$$
\text{Feature}(x) = \sum_{i \in \text{white}} I(i) - \sum_{j \in \text{black}} I(j)
$$

### 3.2 Deep Learning Face Detection
- **SSD (Single Shot Detector)** face models.
- **MTCNN** — Multi-task CNN (face detection + landmark).
- **RetinaFace** — State-of-the-art multi-scale face detector.

```python
# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# DNN Face Detection (Caffe model)
net = cv2.dnn.readNetFromCaffe(prototxt, model)
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
net.setInput(blob)
detections = net.forward()
```

---

## 4. Face Recognition

### 4.1 Pipeline
1. **Detection** — Temukan lokasi wajah.
2. **Alignment** — Normalisasi posisi wajah (berdasarkan landmarks).
3. **Embedding** — Ekstrak fitur vektor dari wajah.
4. **Matching** — Bandingkan embedding dengan database.

### 4.2 Metode Klasik
- **Eigenfaces (PCA)**: Proyeksi ke eigenspace wajah.
- **Fisherfaces (LDA)**: Maksimalkan separasi antar kelas.
- **LBPH (Local Binary Pattern Histogram)**: Fitur tekstur lokal.

### 4.3 Metode Deep Learning
- **FaceNet**: Triplet loss → embedding 128-D, threshold jarak Euclidean.
- **ArcFace**: Angular margin loss untuk embedding lebih diskriminatif.
- **DeepFace**: Framework meta (backend: VGG-Face, Facenet, ArcFace, dll).

$$
L_{\text{triplet}} = \max(0, \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha)
$$

```python
# OpenCV Face Recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces_list, np.array(labels))
label, confidence = recognizer.predict(test_face)
```

---

## 5. Object Recognition dan Classification

### 5.1 Klasik: Bag of Visual Words (BoVW)
1. Deteksi keypoints + deskriptor (SIFT/SURF).
2. Clustering deskriptor → visual vocabulary (k-Means).
3. Histogram frekuensi visual words per gambar.
4. Klasifikasi dengan SVM atau KNN.

### 5.2 Deep Learning Classification
- Arsitektur: ResNet, EfficientNet, ViT (Vision Transformer).
- **Transfer Learning**: Pre-trained ImageNet → fine-tune ke domain spesifik.
- **Feature Extraction**: Gunakan layer konvolusi sebagai feature extractor.

---

## 6. Scene Recognition

### 6.1 Pendekatan
- **GIST Descriptor**: Representasi holistik dari scene.
- **Places CNN**: Model khusus untuk klasifikasi scene (365 kategori).
- **Indoor vs Outdoor**: Tantangan berbeda dalam representasi spatial.

### 6.2 Spatial Pyramid Matching
Membagi gambar menjadi grid bertingkat, menghitung histogram fitur per cell.

$$
K(\mathbf{x}, \mathbf{y}) = \sum_{l=0}^{L} \frac{1}{2^{L-l}} \sum_{i=1}^{D} \min(H_l^{\mathbf{x}}(i), H_l^{\mathbf{y}}(i))
$$

---

## 7. OCR dan Text Recognition

### 7.1 Pipeline
1. **Text Detection**: Lokalisasi area teks (EAST, CRAFT, DBNet).
2. **Text Recognition**: Baca karakter (Tesseract, CRNN, TrOCR).
3. **Post-processing**: Spell checking, formatting.

### 7.2 Tesseract OCR
- Engine open-source dari Google.
- Support 100+ bahasa termasuk Indonesia.
- LSTM-based recognition engine (v4+).

```python
import pytesseract
text = pytesseract.image_to_string(img, lang='eng+ind')
```

### 7.3 Scene Text Detection (EAST)
```python
net = cv2.dnn.readNet("frozen_east_text_detection.pb")
blob = cv2.dnn.blobFromImage(img, 1.0, (320, 320), (123.68, 116.78, 103.94), True, False)
net.setInput(blob)
scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
```

---

## 8. Pedestrian dan Vehicle Detection

### 8.1 HOG + SVM (Klasik)
- Histogram of Oriented Gradients sebagai fitur.
- SVM sebagai classifier.
- Sliding window untuk multi-scale detection.

$$
\text{HOG cell} = \text{histogram orientasi gradien dalam } 8 \times 8 \text{ pixel cell}
$$

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
rects, weights = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8))
```

### 8.2 Deep Learning Detection
- **YOLO**: Real-time detection berbagai kelas (COCO 80 kelas termasuk person, car, truck, bus).
- **SSD MobileNet**: Efisien untuk edge devices.
- **Faster R-CNN**: Akurasi tinggi, lebih lambat.

---

## 9. Gesture dan Action Recognition

### 9.1 Hand Gesture Recognition
- **MediaPipe Hands**: 21 landmark per tangan.
- Klasifikasi gesture dari posisi landmark.
- Aplikasi: sign language, kontrol UI.

### 9.2 Action Recognition
- **Temporal Analysis**: Analisis sekuens frame.
- **3D CNN**: Conv3D untuk spatiotemporal features.
- **Pose-based**: Estimasi pose → klasifikasi gerakan.

```python
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # 21 landmarks per hand
        pass
```

---

## 10. Evaluation Metrics untuk Recognition

### 10.1 Classification Metrics
- **Accuracy**: $\frac{TP+TN}{TP+TN+FP+FN}$
- **Precision**: $\frac{TP}{TP+FP}$
- **Recall**: $\frac{TP}{TP+FN}$
- **F1-Score**: $2 \cdot \frac{P \cdot R}{P + R}$
- **Confusion Matrix**: Visualisasi per-kelas performance.

### 10.2 Detection Metrics
- **IoU (Intersection over Union)**: $\text{IoU} = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}$
- **mAP (mean Average Precision)**: Area under precision-recall curve, di-rata-rata per kelas.
- **AP@50, AP@75**: AP pada IoU threshold tertentu.

### 10.3 Recognition Metrics
- **TAR (True Accept Rate)** vs **FAR (False Accept Rate)**.
- **ROC Curve**: TPR vs FPR.
- **EER (Equal Error Rate)**: Titik di mana FAR = FRR.

---

## 11. Face Embedding dan Distance Metrics

### 11.1 Konsep Face Embedding
Face embedding adalah representasi wajah sebagai vektor berdimensi tinggi (biasanya 128-D atau 512-D) dalam ruang Euclidean. Model seperti FaceNet, ArcFace, dan VGG-Face menghasilkan embedding di mana wajah orang yang sama berdekatan dan wajah orang berbeda berjauhan.

### 11.2 Distance Metrics

**Euclidean Distance:**
$$
d_E(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

**Cosine Distance:**
$$
d_C(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
$$

- **Euclidean**: Sensitif terhadap magnitude vektor, cocok untuk embedding yang sudah dinormalisasi.
- **Cosine**: Mengukur sudut antar vektor, invariant terhadap magnitude.

### 11.3 Verifikasi vs Identifikasi
- **Verifikasi (1:1)**: Bandingkan dua wajah → apakah orang yang sama? Gunakan threshold jarak.
- **Identifikasi (1:N)**: Cari wajah terdekat dalam database → siapa orang ini?

### 11.4 Threshold Selection
- **Intra-class distance**: Jarak antar embedding wajah orang yang sama (harus kecil).
- **Inter-class distance**: Jarak antar embedding wajah orang berbeda (harus besar).
- Threshold optimal: titik di mana distribusi intra-class dan inter-class terpisah dengan baik.

```python
from scipy.spatial.distance import euclidean, cosine

# Euclidean distance
dist_euclidean = euclidean(embedding1, embedding2)

# Cosine distance
dist_cosine = cosine(embedding1, embedding2)

# Verification
is_same = dist_euclidean < threshold
```

---

## 12. Object Tracking untuk Recognition

### 12.1 Konsep Tracking
Object tracking melacak objek yang sama antar frame video, mempertahankan identitas konsisten. Tracking mengurangi kebutuhan deteksi per frame (komputasi mahal) dengan memprediksi posisi objek berdasarkan frame sebelumnya.

### 12.2 Metode Tracking
- **Centroid Tracking**: Melacak pusat bounding box, assign berdasarkan jarak minimum.
- **KCF (Kernelized Correlation Filter)**: Tracking berbasis korelasi di domain Fourier.
- **CSRT (Channel and Spatial Reliability Tracking)**: Lebih akurat dari KCF, lebih lambat.
- **SORT/DeepSORT**: Kombinasi Kalman filter + Hungarian algorithm, dengan appearance features.

### 12.3 Multi-Object Tracking Pipeline
1. **Detection**: Deteksi objek pada frame (setiap N frame).
2. **Prediction**: Prediksi posisi objek pada frame berikutnya.
3. **Association**: Cocokkan deteksi baru dengan track yang ada.
4. **Update**: Perbarui track dengan deteksi yang cocok.
5. **Management**: Register track baru, deregister track yang hilang.

```python
# OpenCV Multi-tracker
trackers = cv2.legacy.MultiTracker_create()
for bbox in initial_detections:
    tracker = cv2.legacy.TrackerCSRT_create()
    trackers.add(tracker, frame, bbox)

# Update tracking
success, boxes = trackers.update(new_frame)
```

### 12.4 Tantangan Tracking
- **Occlusion**: Objek terhalang sementara.
- **ID Switch**: Identitas tertukar saat objek berdekatan.
- **Scale Change**: Ukuran objek berubah (mendekat/menjauh).
- **Re-identification**: Mengenali kembali objek setelah hilang.

---

## 13. Recognition Pipeline Terintegrasi

### 13.1 End-to-End Pipeline
Pipeline recognition lengkap mengintegrasikan:
1. **Input**: Gambar atau video stream.
2. **Detection**: Lokalisasi objek/wajah (Haar, DNN, HOG).
3. **Preprocessing**: Alignment, normalisasi, resize.
4. **Feature Extraction**: Embedding atau descriptor.
5. **Recognition/Classification**: Matching atau klasifikasi.
6. **Post-processing**: NMS, filtering, confidence thresholding.
7. **Evaluation**: Metrik performa (accuracy, mAP, ROC).

### 13.2 Pertimbangan Desain
- **Modularitas**: Setiap komponen dapat diganti independen.
- **Latency vs Accuracy**: Trade-off antara kecepatan dan akurasi.
- **Scalability**: Performa saat database bertambah besar.
- **Error Propagation**: Error di tahap awal mempengaruhi seluruh pipeline.

---

## 14. Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Ed., Chapter 6 — Recognition.
2. Viola, P. & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*. CVPR.
3. Schroff, F. et al. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR.
4. Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
5. Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.
6. Smith, R. (2007). *An Overview of the Tesseract OCR Engine*. ICDAR.
7. Zhou, X. et al. (2017). *EAST: An Efficient and Accurate Scene Text Detector*. CVPR.
8. Zhang, F. et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. CVPR Workshop.
9. Deng, J. et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR.
10. Turk, M. & Pentland, A. (1991). *Eigenfaces for Recognition*. J. Cognitive Neuroscience.
