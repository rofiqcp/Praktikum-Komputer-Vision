# MODUL 9: DEEP LEARNING UNTUK KOMPUTER VISION

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 5: Deep Learning.**

---

## 9.1 Pendahuluan

Deep learning telah merevolusi computer vision sejak AlexNet memenangkan ImageNet Challenge 2012. Jaringan saraf dalam (deep neural networks) mampu belajar representasi fitur secara otomatis dari data, menggantikan pendekatan hand-crafted features yang mendominasi sebelumnya.

---

## 9.2 Artificial Neural Networks (ANN)

### Perceptron dan Multi-Layer Perceptron (MLP)
Neuron: $y = \sigma(\mathbf{w}^T\mathbf{x} + b)$

Fungsi aktivasi:
- **ReLU**: $\sigma(z) = \max(0, z)$ — paling populer, mengatasi vanishing gradient.
- **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$ — output [0,1], cocok untuk probabilitas.
- **Tanh**: $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ — output [-1,1], zero-centered.
- **LeakyReLU**: $\sigma(z) = \max(\alpha z, z)$ — mengatasi dying ReLU problem.
- **Softmax** (output layer): $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ — distribusi probabilitas multi-kelas.

### Training
- **Forward pass**: Hitung prediksi dari input ke output.
- **Loss function**: Cross-entropy (klasifikasi), MSE (regresi).
- **Backpropagation**: Hitung gradien loss terhadap setiap weight menggunakan chain rule.
- **Optimizer**: SGD, SGD+Momentum, Adam, RMSProp.
- **Learning rate scheduling**: StepLR, CosineAnnealing, ReduceLROnPlateau.

---

## 9.3 Convolutional Neural Networks (CNN)

### Arsitektur CNN
```
Input → [Conv → ReLU → Pool]×N → Flatten → FC → Output
```

### Konvolusi pada CNN
Filter (kernel) yang shared secara spasial:
$$y_{ij} = \sum_k \sum_l w_{kl} \cdot x_{i+k, j+l} + b$$

Parameter konvolusi:
- **Kernel size**: Ukuran filter (3×3, 5×5, 7×7).
- **Stride**: Langkah pergeseran filter.
- **Padding**: Penambahan piksel di tepi (valid, same).
- **Channels**: Jumlah filter = jumlah output feature maps.

### Komponen CNN
- **Convolutional Layer**: Ekstraksi fitur lokal dengan shared weights.
- **Pooling Layer**: Downsampling (Max Pool: ambil nilai terbesar, Average Pool: rata-rata).
- **Batch Normalization**: Normalisasi output per mini-batch, stabilisasi training.
- **Dropout**: Menonaktifkan neuron secara random untuk regularisasi.
- **Fully Connected Layer**: Klasifikasi akhir.

### Arsitektur Terkenal
| Model | Tahun | Keunggulan | Parameter |
|-------|-------|------------|-----------|
| **LeNet-5** | 1998 | Pionir CNN, digit recognition | ~60K |
| **AlexNet** | 2012 | GPU training, ReLU, dropout | ~60M |
| **VGG** | 2014 | Uniform 3×3 filters, sangat dalam | ~138M |
| **GoogLeNet/Inception** | 2014 | Inception module, multi-scale | ~5M |
| **ResNet** | 2015 | Skip connections, 152+ layers | ~25M |
| **MobileNet** | 2017 | Depthwise separable conv, lightweight | ~3.4M |
| **EfficientNet** | 2019 | Compound scaling (depth+width+resolution) | ~5.3M |

---

## 9.4 Transfer Learning

Menggunakan model yang telah di-train pada dataset besar (ImageNet) sebagai starting point:

1. **Training from Scratch**: Melatih semua layer dari awal, butuh data besar.
2. **Feature Extraction**: Freeze seluruh layer kecuali classifier terakhir, train hanya FC.
3. **Fine-tuning**: Unfreeze beberapa layer terakhir dan train dengan learning rate kecil.

Keuntungan: Memerlukan data lebih sedikit, training lebih cepat, performa lebih baik.

---

## 9.5 Data Augmentation

Teknik memperbanyak data training secara artifisial:
- **Geometric**: Flip, rotation, crop, scale, translate.
- **Photometric**: Brightness, contrast, hue, saturation, noise.
- **Advanced**: Cutout, Mixup, CutMix, RandAugment.

Tujuan: Mencegah overfitting, meningkatkan generalisasi model.

---

## 9.6 Object Detection

### Arsitektur
- **Two-stage**: R-CNN → Fast R-CNN → Faster R-CNN (Region Proposal + Classification).
- **One-stage**: YOLO, SSD, RetinaNet (langsung prediksi box + class).

### Sliding Window dan Image Pyramid
Pendekatan klasik: scan gambar dengan window berbagai ukuran pada berbagai skala.
- **Image Pyramid**: Resize gambar ke beberapa level (multi-scale).
- **Non-Maximum Suppression (NMS)**: Menghilangkan deteksi duplikat.

### HOG (Histogram of Oriented Gradients)
1. Hitung gradient magnitude dan orientasi setiap piksel.
2. Bagi gambar ke cells (8×8 piksel), buat histogram orientasi.
3. Normalisasi blok (2×2 cells).
4. Gabungkan menjadi vektor fitur → + SVM classifier.

### YOLO (You Only Look Once)
- Membagi gambar ke grid S×S.
- Setiap cell memprediksi B bounding boxes dan C class probabilities.
- Sangat cepat, cocok untuk real-time.
- Anchor boxes: predefined box shapes untuk berbagai aspek rasio.

### Metrics Deteksi
- **IoU (Intersection over Union)**: $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$
- **mAP (mean Average Precision)**: Rata-rata AP per kelas.
- **FPS (Frames Per Second)**: Kecepatan inferensi.

---

## 9.7 Semantic Segmentation

Klasifikasi per-piksel: setiap piksel diberi label kelas.

Model: FCN, U-Net, DeepLab, PSPNet, SegFormer.

Loss: Cross-entropy per piksel, Dice loss.

Pendekatan manual:
- **Color-based**: Segmentasi berdasarkan range warna HSV.
- **K-Means**: Clustering piksel berdasarkan warna.
- **Thresholding + Morphology**: Binary segmentasi dengan operasi erode/dilate.

---

## 9.8 Instance Segmentation

Kombinasi detection + segmentation: mendeteksi setiap instance objek dan segmentasi mask-nya.

Perbedaan dari semantic segmentation:
- **Semantic**: Semua objek kelas yang sama diberi label sama.
- **Instance**: Setiap objek individual dideteksi dan di-mask terpisah.

Model: Mask R-CNN, YOLACT, SOLOv2.

---

## 9.9 Loss Functions

### Cross-Entropy Loss
$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

### Mean Squared Error (MSE)
$$L = \frac{1}{n}\sum_{i}(y_i - \hat{y}_i)^2$$

### Binary Cross-Entropy (BCE)
$$L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

### Dice Loss
$$L_{Dice} = 1 - \frac{2|A \cap B|}{|A| + |B|}$$

### Focal Loss
Mengatasi class imbalance:
$$L_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

| Loss Function | Kegunaan | Kelebihan |
|--------------|----------|----------|
| Cross-Entropy | Klasifikasi | Standar, stabil |
| MSE | Regresi | Sensitif terhadap outlier |
| Dice | Segmentasi | Baik untuk class imbalance |
| Focal | Deteksi | Mengatasi easy vs hard examples |

---

## 9.10 Optimizers

### SGD (Stochastic Gradient Descent)
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

### SGD with Momentum
$$v_t = \beta v_{t-1} + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

### RMSProp
$$s_t = \beta s_{t-1} + (1-\beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} g_t$$

### Adam (Adaptive Moment Estimation)
Kombinasi momentum dan adaptive learning rate:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

| Optimizer | Kelebihan | Kekurangan |
|-----------|-----------|------------|
| SGD | Sederhana, generalisasi baik | Konvergensi lambat |
| Momentum | Lebih cepat, melewati local minima | Perlu tuning β |
| RMSProp | Adaptif, baik untuk sparse gradient | Perlu tuning decay |
| Adam | Cepat konvergen, adaptif | Bisa kurang generalisasi |

---

## 9.11 Batch Normalization dan Dropout

### Batch Normalization
Normalisasi aktivasi per mini-batch:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

Keuntungan: training lebih stabil, memungkinkan learning rate lebih tinggi, mengurangi ketergantungan pada inisialisasi weight.

### Dropout
Secara random menonaktifkan neuron selama training dengan probabilitas $p$:
- **Training**: Setiap neuron di-drop dengan probabilitas $p$.
- **Inference**: Semua neuron aktif, output di-scale dengan $(1-p)$.
- **Inverted Dropout**: Scale saat training oleh $\frac{1}{1-p}$, inference tidak perlu scale.

Efek: Mencegah overfitting, memaksa representasi redundan.

---

## 9.12 Model Evaluation Metrics

### Metrik Klasifikasi
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$ — seberapa tepat prediksi positif.
- **Recall**: $\frac{TP}{TP + FN}$ — seberapa lengkap deteksi positif.
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$

### Kurva Evaluasi
- **ROC Curve**: Plot TPR vs FPR pada berbagai threshold.
- **AUC (Area Under Curve)**: Luas di bawah ROC, semakin besar semakin baik.
- **Precision-Recall Curve**: Fokus pada performa kelas positif.

### Metrik Segmentasi
- **Pixel Accuracy**: Persentase piksel yang benar.
- **Mean IoU (mIoU)**: Rata-rata IoU per kelas.
- **Dice Coefficient**: $\frac{2|A \cap B|}{|A| + |B|}$

---

## 9.13 ONNX Export dan Deployment

### Pipeline Deployment
```
Training (PyTorch/TF) → Export ONNX → Optimasi → Runtime → Deployment
```

### Optimasi Model
- **Quantization**: FP32 → FP16 → INT8 (mengurangi ukuran 4x, mempercepat inferensi).
- **Pruning**: Menghapus koneksi/neuron yang tidak penting.
- **Knowledge Distillation**: Melatih model kecil (student) dari model besar (teacher).

### Runtime Options
| Runtime | Platform | Kelebihan |
|---------|----------|----------|
| ONNX Runtime | Cross-platform | Universal, mudah |
| TensorRT | NVIDIA GPU | Optimasi maksimal GPU |
| OpenVINO | Intel CPU/GPU | Optimasi hardware Intel |
| TFLite | Mobile | Ringan untuk Android/iOS |
| OpenCV DNN | Cross-platform | Ringan, tanpa framework berat |

---

## 9.14 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| ANN/MLP | Jaringan saraf dasar, perceptron, backpropagation |
| CNN | Konvolusi + pooling, ekstraksi fitur otomatis |
| Arsitektur | LeNet → AlexNet → VGG → ResNet → MobileNet → EfficientNet |
| Transfer Learning | Memanfaatkan model pre-trained untuk domain baru |
| Data Augmentation | Memperbanyak data secara artifisial |
| Object Detection | Sliding window, HOG+SVM, YOLO |
| Segmentation | Semantic (per-piksel), Instance (per-objek) |
| Loss Functions | CE, MSE, Dice, Focal Loss |
| Optimizers | SGD, Momentum, RMSProp, Adam |
| Batch Norm & Dropout | Stabilisasi training, regularisasi |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, mAP |
| Deployment | ONNX, quantization, OpenCV DNN |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. **Chapter 5: Deep Learning** (pp. 267–350).
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, MIT Press.
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
4. Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." *CVPR*.
5. Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*.
6. Rosebrock, A. (2017). *Deep Learning for Computer Vision with Python*.
7. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd Ed.
