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

## 5.10 Ringkasan

| Konsep | Penjelasan |
|--------|------------|
| ANN/MLP | Jaringan saraf dasar, backpropagation |
| CNN | Konvolusi + pooling, ekstraksi fitur otomatis |
| Transfer Learning | Memanfaatkan model pre-trained |
| Data Augmentation | Memperbanyak data training secara artifisial |
| Object Detection | YOLO, Faster R-CNN — lokasi + kelas objek |
| Segmentation | Semantic (per-piksel), Instance (per-objek) |
| ONNX | Format model universal untuk deployment |

---

## Referensi

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd Edition. **Chapter 5: Deep Learning** (pp. 267–350).
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, MIT Press.
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
4. Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." *CVPR*.
5. Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*.
6. PyTorch Documentation. https://pytorch.org/docs/
7. TensorFlow/Keras Documentation. https://www.tensorflow.org/
