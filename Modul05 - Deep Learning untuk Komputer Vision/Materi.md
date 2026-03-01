# MODUL 5: DEEP LEARNING UNTUK KOMPUTER VISION

## Referensi Utama
**Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. Chapter 5: Deep Learning.**

---

## 5.1 Pendahuluan

Deep learning telah merevolusi computer vision sejak AlexNet memenangkan ImageNet Challenge 2012. Jaringan saraf dalam (deep neural networks) mampu belajar representasi fitur secara otomatis dari data, menggantikan pendekatan hand-crafted features yang mendominasi sebelumnya.

---

## 5.2 Artificial Neural Networks (ANN)

### Perceptron dan Multi-Layer Perceptron (MLP)
Neuron: $y = \sigma(\mathbf{w}^T\mathbf{x} + b)$

Fungsi aktivasi:
- **ReLU**: $\sigma(z) = \max(0, z)$ — paling populer.
- **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Softmax** (output layer): $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

### Training
- **Forward pass**: Hitung prediksi.
- **Loss function**: Cross-entropy (klasifikasi), MSE (regresi).
- **Backpropagation**: Hitung gradien loss terhadap setiap weight.
- **Optimizer**: SGD, Adam, AdamW.
- **Learning rate scheduling**: StepLR, CosineAnnealing.

---

## 5.3 Convolutional Neural Networks (CNN)

### Arsitektur CNN
```
Input → [Conv → ReLU → Pool]×N → Flatten → FC → Output
```

### Konvolusi pada CNN
Filter (kernel) yang shared secara spasial:
$$y_{ij} = \sum_k \sum_l w_{kl} \cdot x_{i+k, j+l} + b$$

### Komponen CNN
- **Convolutional Layer**: Ekstraksi fitur lokal.
- **Pooling Layer**: Downsampling (Max Pool, Average Pool).
- **Batch Normalization**: Stabilisasi training.
- **Dropout**: Regularisasi untuk mencegah overfitting.
- **Fully Connected Layer**: Klasifikasi akhir.

### Arsitektur Terkenal
| Model | Tahun | Keunggulan |
|-------|-------|------------|
| **LeNet-5** | 1998 | Pionir CNN untuk digit recognition |
| **AlexNet** | 2012 | GPU training, ReLU, dropout |
| **VGG** | 2014 | Uniform 3×3 filters, sangat dalam |
| **GoogLeNet/Inception** | 2014 | Inception module, multi-scale |
| **ResNet** | 2015 | Skip connections, 152+ layers |
| **MobileNet** | 2017 | Depthwise separable conv, lightweight |
| **EfficientNet** | 2019 | Compound scaling |

---

## 5.4 Transfer Learning

Menggunakan model yang telah di-train pada dataset besar (ImageNet) sebagai starting point:

1. **Feature Extraction**: Freeze seluruh layer kecuali classifier terakhir.
2. **Fine-tuning**: Unfreeze beberapa layer terakhir dan train dengan learning rate kecil.

Keuntungan: Memerlukan data lebih sedikit, training lebih cepat, performa lebih baik.

---

## 5.5 Data Augmentation

Teknik memperbanyak data training secara artifisial:
- **Geometric**: Flip, rotation, crop, scale, translate.
- **Photometric**: Brightness, contrast, hue, saturation, noise.
- **Advanced**: Cutout, Mixup, CutMix, RandAugment.

---

## 5.6 Object Detection

### Arsitektur
- **Two-stage**: R-CNN → Fast R-CNN → Faster R-CNN (Region Proposal + Classification).
- **One-stage**: YOLO, SSD, RetinaNet (langsung prediksi box + class).

### YOLO (You Only Look Once)
- Membagi gambar ke grid S×S.
- Setiap cell memprediksi B bounding boxes dan C class probabilities.
- Sangat cepat, cocok untuk real-time.

### Metrics
- **IoU (Intersection over Union)**: $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$
- **mAP (mean Average Precision)**: Rata-rata AP per kelas.
- **FPS (Frames Per Second)**: Kecepatan inferensi.

---

## 5.7 Semantic Segmentation

Klasifikasi per-piksel: setiap piksel diberi label kelas.

Model: FCN, U-Net, DeepLab, PSPNet, SegFormer.

Loss: Cross-entropy per piksel, Dice loss.

---

## 5.8 Instance Segmentation

Kombinasi detection + segmentation: mendeteksi setiap instance objek dan segmentasi mask-nya.

Model: Mask R-CNN, YOLACT, SOLOv2.

---

## 5.9 Model Deployment

### ONNX (Open Neural Network Exchange)
Format universal untuk model DNN. Export dari PyTorch/Keras → inferensi di berbagai runtime.

### OpenCV DNN Module
`cv2.dnn.readNet()` — load model pre-trained untuk inferensi tanpa framework DL penuh.

### Optimasi Inferensi
- Quantization (FP32 → INT8).
- Pruning.
- Knowledge distillation.
- TensorRT, OpenVINO.

---

## 5.10 HOG (Histogram of Oriented Gradients)

### Konsep HOG
HOG adalah fitur deskriptor yang digunakan untuk deteksi objek, khususnya deteksi pejalan kaki. Proses:
1. **Gradient computation**: Hitung magnitude dan orientasi gradien setiap piksel.
2. **Cell histograms**: Bagi gambar ke cells (8×8 piksel), buat histogram orientasi.
3. **Block normalization**: Normalisasi histogram dalam blok (2×2 cells) untuk invariansi pencahayaan.
4. **Feature vector**: Gabungkan semua histogram menjadi satu vektor fitur.

### HOG + SVM vs Deep Learning
| Aspek | HOG + SVM | Deep Learning |
|-------|-----------|---------------|
| Training data | Sedikit | Banyak |
| Kecepatan | Cepat (CPU) | Memerlukan GPU |
| Akurasi | Baik untuk objek rigid | Lebih baik untuk variasi tinggi |
| Generalisasi | Terbatas | Sangat baik |

---

## 5.11 Loss Functions

### Cross-Entropy Loss
Untuk klasifikasi:
$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

### Mean Squared Error (MSE)
Untuk regresi:
$$L = \frac{1}{n}\sum_{i}(y_i - \hat{y}_i)^2$$

### Dice Loss
Untuk segmentasi, mengukur overlap:
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

## 5.12 Optimizers

### SGD (Stochastic Gradient Descent)
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

### SGD with Momentum (SGDM)
$$v_t = \beta v_{t-1} + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

### Adam (Adaptive Moment Estimation)
Kombinasi momentum dan adaptive learning rate:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### Perbandingan Optimizer
| Optimizer | Kelebihan | Kekurangan |
|-----------|-----------|------------|
| SGD | Sederhana, generalisasi baik | Konvergensi lambat |
| SGDM | Lebih cepat dari SGD | Perlu tuning momentum |
| Adam | Konvergensi cepat, adaptif | Bisa kurang generalisasi |

---

## 5.13 Batch Normalization dan Dropout

### Batch Normalization
Normalisasi aktivasi per mini-batch:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

Keuntungan: training lebih stabil, memungkinkan learning rate lebih tinggi, mengurangi ketergantungan pada inisialisasi weight.

### Dropout
Secara random menonaktifkan neuron selama training dengan probabilitas $p$:
- **Training**: Setiap neuron di-drop dengan probabilitas $p$.
- **Inference**: Semua neuron aktif, output di-scale dengan $(1-p)$.

Efek: Regularisasi yang mencegah overfitting, memaksa jaringan belajar fitur yang lebih robust dan tidak bergantung pada neuron tertentu.

---

## 5.14 Model Evaluation Metrics

### Metrik Klasifikasi
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$ — seberapa tepat prediksi positif.
- **Recall**: $\frac{TP}{TP + FN}$ — seberapa lengkap deteksi positif.
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$

### Metrik Deteksi Objek
- **IoU (Intersection over Union)**: $\frac{|A \cap B|}{|A \cup B|}$
- **mAP (mean Average Precision)**: Rata-rata AP di semua kelas pada berbagai IoU threshold.
- **Precision-Recall Curve**: Plot precision vs recall pada berbagai confidence threshold.

### Metrik Segmentasi
- **Pixel Accuracy**: Persentase piksel yang benar.
- **Mean IoU (mIoU)**: Rata-rata IoU per kelas.
- **Dice Coefficient**: $\frac{2|A \cap B|}{|A| + |B|}$

---

## 5.15 ONNX Export dan Deployment

### Pipeline Deployment
```
Training (PyTorch/TF) → Export ONNX → Optimasi → Runtime → Deployment
```

### ONNX Export
```python
# PyTorch
torch.onnx.export(model, dummy_input, "model.onnx")

# TensorFlow/Keras
import tf2onnx
tf2onnx.convert.from_keras(model, output_path="model.onnx")
```

### Optimasi Model untuk Deployment
- **Quantization**: FP32 → FP16 → INT8 (mengurangi ukuran dan mempercepat inferensi).
- **Pruning**: Menghapus koneksi/neuron yang tidak penting.
- **Knowledge Distillation**: Melatih model kecil (student) dari model besar (teacher).

### Runtime Options
| Runtime | Platform | Kelebihan |
|---------|----------|----------|
| ONNX Runtime | Cross-platform | Universal, mudah digunakan |
| TensorRT | NVIDIA GPU | Optimasi maksimal untuk GPU |
| OpenVINO | Intel CPU/GPU | Optimasi untuk hardware Intel |
| TFLite | Mobile | Ringan untuk Android/iOS |

---

## 5.16 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| ANN/MLP | Jaringan saraf dasar, backpropagation |
| CNN | Konvolusi + pooling, ekstraksi fitur otomatis |
| Transfer Learning | Memanfaatkan model pre-trained |
| Data Augmentation | Memperbanyak data training secara artifisial |
| Object Detection | YOLO, Faster R-CNN — lokasi + kelas objek |
| Segmentation | Semantic (per-piksel), Instance (per-objek) |
| HOG | Fitur deskriptor klasik untuk deteksi objek |
| Loss Functions | Cross-Entropy, MSE, Dice, Focal Loss |
| Optimizers | SGD, SGDM, Adam — mengoptimalkan proses training |
| Batch Norm & Dropout | Teknik regularisasi dan stabilisasi training |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, mAP, IoU |
| ONNX & Deployment | Format universal untuk deployment model |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. **Chapter 5: Deep Learning** (pp. 267–350).
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, MIT Press.
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
4. Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." *CVPR*.
5. Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*.
6. PyTorch Documentation. https://pytorch.org/docs/
7. TensorFlow/Keras Documentation. https://www.tensorflow.org/
