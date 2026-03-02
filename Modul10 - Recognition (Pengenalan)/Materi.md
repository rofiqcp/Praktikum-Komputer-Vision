# MODUL 10: RECOGNITION (PENGENALAN)

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 6: Recognition.**
**Mastering OpenCV 4 with Python (Fernández Villán, 2019). Ch.6-7: Face Detection & Recognition.**
**Machine Learning for OpenCV (Beyeler, 2017). Ch.3-8: ML Classifiers.**

---

## 10.1 Pendahuluan

Recognition (pengenalan) adalah kemampuan sistem komputer untuk mengidentifikasi dan mengklasifikasikan objek, wajah, teks, atau scene dalam gambar. Ini merupakan salah satu tujuan utama computer vision.

### Taxonomi Recognition
- **Classification**: Menentukan kategori gambar (kucing/anjing).
- **Detection**: Melokalisasi objek dalam bounding box.
- **Segmentation**: Memetakan setiap piksel ke kelas.
- **Identification**: Mengenali identitas spesifik (siapa orang ini?).

### Instance vs Category Recognition
- **Instance recognition**: Mengenali objek spesifik (wajah Andi).
- **Category recognition**: Mengenali kelas objek (ini adalah wajah manusia).

---

## 10.2 Face Detection

### Viola-Jones (Haar Cascade)
Metode klasik yang menggunakan fitur Haar-like:

$$f = \sum_{(x,y) \in \text{white}} I(x,y) - \sum_{(x,y) \in \text{black}} I(x,y)$$

Komponen:
- **Integral image** untuk perhitungan cepat.
- **AdaBoost** untuk seleksi fitur terbaik.
- **Cascade classifier**: serangkaian stage, reject cepat pada stage awal.

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
```

### DNN Face Detection
Model Caffe/TensorFlow via OpenCV DNN module:
- **Akurasi**: >99% (WIDER FACE benchmark).
- **Kecepatan**: ~20-50 FPS (GPU).
- **cv2.dnn.readNetFromCaffe()** untuk deployment.

---

## 10.3 Face Recognition

### Pipeline Face Recognition
```
Input → Face Detection → Face Alignment → Feature Extraction → Matching → Identity
```

### Eigenfaces (PCA)
Representasikan wajah sebagai kombinasi linear eigenfaces (komponen utama):

$$\mathbf{x} \approx \bar{\mathbf{x}} + \sum_{i=1}^{k} w_i \mathbf{u}_i$$

- $\bar{\mathbf{x}}$: mean face
- $\mathbf{u}_i$: eigenface ke-$i$
- $w_i$: bobot (proyeksi)

### LBPH (Local Binary Pattern Histogram)
Tekstur lokal yang robust terhadap pencahayaan:

$$LBP(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p$$

di mana $s(z) = 1$ jika $z \geq 0$, else $0$.

### Face Embedding (FaceNet/ArcFace)
Representasikan wajah sebagai vektor berdimensi tinggi:
- **FaceNet**: 128-d embedding, triplet loss.
- **ArcFace**: 512-d, additive angular margin loss.
- Jarak Euclidean kecil → wajah sama.
- Jarak Cosine: $\text{sim} = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|}$

---

## 10.4 Face Landmark Detection

68 titik landmark standar (dlib model):
- **Jawline**: titik 0-16
- **Alis**: titik 17-26
- **Hidung**: titik 27-35
- **Mata**: titik 36-47
- **Mulut**: titik 48-67

Aplikasi: face alignment, expression analysis, face morphing, gaze estimation.

---

## 10.5 Optical Character Recognition (OCR)

### Pipeline OCR
```
Input → Preprocessing → Text Detection → Character Segmentation → Recognition → Output
```

### Preprocessing untuk OCR
1. **Binarisasi**: Otsu, Adaptive Threshold.
2. **Denoising**: Gaussian Blur, Median, fastNlMeansDenoising.
3. **Deskew**: Perbaiki kemiringan teks.
4. **Resize**: Optimal 300 DPI untuk Tesseract.

### Tesseract OCR
Engine OCR open-source:
- **V4+**: LSTM-based untuk akurasi lebih tinggi.
- **Multi-bahasa**: 100+ bahasa.

---

## 10.6 Scene Text Detection

### MSER (Maximally Stable Extremal Regions)
Mendeteksi region dengan intensitas stabil pada berbagai threshold. Cocok untuk karakter teks (kontras tinggi).

### Morphology-based
Pipeline: Gradient → Threshold → Dilasi horizontal → Contour filtering.

---

## 10.7 HOG Pedestrian Detection

Histogram of Oriented Gradients (HOG):
1. **Gradient**: Magnitude $m = \sqrt{g_x^2 + g_y^2}$, Orientasi $\theta = \arctan(g_y/g_x)$.
2. **Cell histogram**: 8×8 piksel, 9 bins orientasi (0°-180°).
3. **Block normalization**: 2×2 cells, L2 normalization.
4. **Classifier**: Linear SVM.

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(img)
```

---

## 10.8 Object Classification

### Bag of Visual Words (BoVW)
Analog Bag of Words pada NLP:
1. **Ekstrak fitur lokal** (SIFT/ORB) dari training images.
2. **K-Means clustering** → Visual dictionary (K visual words).
3. **Hitung histogram** per gambar: frekuensi visual word.
4. **Klasifikasi** (SVM/KNN) berdasarkan histogram.

### Scene Recognition
Fitur untuk scene: histogram warna, LBP tekstur, GIST descriptor, Places CNN features.

---

## 10.9 Hand Gesture Recognition

### Pipeline Gesture Recognition
1. **Segmentasi tangan**: Warna kulit pada HSV.
2. **Convex Hull**: Boundary terluar tangan.
3. **Convexity Defects**: Titik terjauh dari hull ke kontur (celah antar jari).
4. **Hitung jari**: Sudut defect < 90° = jari.

---

## 10.10 Metrik Evaluasi Classification

### Confusion Matrix
|  | Predicted Positive | Predicted Negative |
|--|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

### Metrik
- **Accuracy**: $\frac{TP+TN}{TP+TN+FP+FN}$
- **Precision**: $\frac{TP}{TP+FP}$ (berapa prediksi positif yang benar)
- **Recall**: $\frac{TP}{TP+FN}$ (berapa data positif yang terdeteksi)
- **F1-Score**: $\frac{2 \cdot P \cdot R}{P + R}$ (harmonic mean)

---

## 10.11 Metrik Evaluasi Detection

### IoU (Intersection over Union)
$$IoU = \frac{|A \cap B|}{|A \cup B|}$$

Threshold standar:
- IoU ≥ 0.5: mAP@50
- IoU ≥ 0.75: mAP@75

### Average Precision (AP)
Area under Precision-Recall curve.

---

## 10.12 ROC Curve dan AUC

### ROC (Receiver Operating Characteristic)
Plot TPR vs FPR pada berbagai threshold:
- **TPR** (True Positive Rate) = Recall
- **FPR** (False Positive Rate) = $\frac{FP}{FP+TN}$
- **AUC** = Area Under ROC Curve. AUC = 1.0 ideal, 0.5 = random.

### Score Distribution
- **Genuine scores**: skor antara wajah orang yang sama.
- **Impostor scores**: skor antara wajah orang berbeda.
- Distribusi terpisah → recognition baik.

---

## 10.13 Face Tracking

### Multi-object Tracking
Pipeline: Detection per frame → Assignment ID berdasarkan jarak center.
Tantangan: oklusi, entry/exit objek baru, ID switch.

---

## 10.14 Recognition Pipeline End-to-End

```
Detect → Align → Embed → Match → Result
```

1. **Detect**: Haar Cascade / DNN (lokalisasi wajah).
2. **Align**: Crop, resize, normalisasi pencahayaan.
3. **Embed**: Ekstrak fitur vektor (PCA, LBPH, deep embedding).
4. **Match**: Bandingkan dengan database (Euclidean/Cosine distance).
5. **Result**: Identitas + confidence score.
