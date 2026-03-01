# NotebookLM Prompts — Modul 5: Deep Learning untuk Komputer Vision

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet)

Buat 15 slide presentasi akademik Modul 5: Deep Learning untuk Komputer Vision. Referensi Szeliski (2022) Ch.5. Tiap slide ~500 kata, sertakan diagram arsitektur dan kode Python.

**Slide 1** — Judul "Modul 5: Deep Learning untuk Komputer Vision", subtitle "Dari Piksel ke Prediksi dengan Jaringan Saraf Dalam", ilustrasi CNN pipeline, referensi Szeliski Ch.5.

**Slide 2** — Sejarah: Perceptron (1958) → MLP → LeNet (1998) → AlexNet (2012, ImageNet) → VGG/GoogLeNet/ResNet (2014–2015) → EfficientNet (2019) → ViT (2020). Diagram timeline.

**Slide 3** — ANN: Neuron y=σ(wᵀx+b). Aktivasi: ReLU=max(0,z), Sigmoid=1/(1+e⁻ᶻ), Softmax. Loss: Cross-entropy (klasifikasi), MSE (regresi). Forward pass, backpropagation, optimizer SGD/Adam.

**Slide 4** — CNN: filter shared secara spasial, y_ij=ΣΣw_kl·x_{i+k,j+l}+b. Komponen: Conv→ReLU→Pool→BatchNorm→Dropout→FC. Max Pool vs Average Pool. Diagram feature map.

**Slide 5** — Arsitektur terkenal: LeNet-5 (digit, 1998), AlexNet (GPU+ReLU+Dropout, 2012), VGG (3×3 uniform, 2014), GoogLeNet/Inception (multi-scale module, 2014), ResNet (skip connection, 2015), MobileNet (depthwise separable conv, 2017), EfficientNet (compound scaling, 2019). Tabel perbandingan.

**Slide 6** — Transfer Learning: Feature Extraction (freeze semua kecuali FC terakhir) vs Fine-tuning (unfreeze beberapa layer terakhir, lr kecil). Keuntungan: data lebih sedikit, konvergen lebih cepat. Diagram: ImageNet → domain spesifik.

**Slide 7** — Data Augmentation: Geometric (flip, rotation, crop, scale, translate). Photometric (brightness, contrast, hue, saturation, noise). Advanced: Cutout, Mixup, CutMix, RandAugment. Kode torchvision.transforms.

**Slide 8** — Object Detection: Two-stage (R-CNN→Fast R-CNN→Faster R-CNN). One-stage (YOLO, SSD, RetinaNet). YOLO: grid S×S, prediksi B bbox + C class per cell. IoU=|A∩B|/|A∪B|. mAP. Percobaan 7–8.

**Slide 9** — Semantic Segmentation: klasifikasi per-piksel. Model: FCN, U-Net (encoder-decoder + skip), DeepLab (atrous conv), SegFormer. Loss: cross-entropy per piksel + Dice loss. Instance Segmentation: Mask R-CNN = Faster R-CNN + mask head. Percobaan 9–10.

**Slide 10** — Percobaan 1–2: OpenCV DNN klasifikasi. cv2.dnn.readNet(), blobFromImage(scalefactor, size, mean), net.setInput(blob), net.forward(). Top-5 prediksi ImageNet 1000 kelas. Perbandingan MobileNetV2/ResNet50/GoogLeNet: akurasi vs FPS vs ukuran.

**Slide 11** — Percobaan 3–4: Build CNN PyTorch (nn.Conv2d→ReLU→MaxPool→FC), training loop CIFAR-10. Keras/TensorFlow Sequential API. Visualisasi arsitektur, feature maps per layer, aktivasi. Layer-by-layer diagram.

**Slide 12** — Percobaan 5–8: Fungsi aktivasi (plot ReLU/Sigmoid/Tanh). Operasi konvolusi manual (filter Sobel, Gaussian). Data augmentation pipeline (sebelum/sesudah, 8 teknik). Transfer learning freeze/unfreeze layers, akurasi perbandingan.

**Slide 13** — Percobaan 9–12: Visualisasi backpropagation (gradient flow, loss descent). Deteksi sliding window (multi-scale, NMS). Deteksi dengan HOG+SVM (cv2.HOGDescriptor). YOLO grid overlay: anchor boxes, IoU, NMS post-processing.

**Slide 14** — Percobaan 13–16: Semantic segmentation manual (threshold + morphology + overlay). Instance segmentation (connected components + watershed). Loss functions plot: MSE vs Cross-Entropy vs Dice vs Focal. Optimizer trajectory: SGD vs SGDM vs Adam convergence.

**Slide 15** — Percobaan 17–20: BatchNorm (distribusi aktivasi sebelum/sesudah) + Dropout (efek regularisasi). Evaluation metrics: confusion matrix, PR curve, ROC, mAP, IoU. ONNX export + runtime inferensi + quantization. Proyek klasifikasi bentuk end-to-end. Setup: torch, tensorflow, ultralytics, onnxruntime.

---

## PROMPT 2 — Slide 16–30 (Materi + Project + TugasVideo)

Lanjutkan slide Modul 5, Slide 16–30. Slide 16–25: analisis, rekap, koneksi, kuis. Slide 26–30: Project dan Tugas Video. Tiap slide ~500 kata.

**Slide 16** — Rekap percobaan 1–10: OpenCV DNN klasifikasi, perbandingan model, CNN PyTorch, CNN Keras, aktivasi, konvolusi, augmentasi, transfer learning, backpropagation, sliding window. Grid thumbnail output. Tabel nama file, konsep kunci, framework.

**Slide 17** — Rekap percobaan 11–20: HOG detection, YOLO grid, semantic segmentation, instance segmentation, loss functions, optimizer, BatchNorm+Dropout, evaluasi metrik, ONNX deployment, proyek end-to-end. Tabel fungsi utama dan library tiap percobaan.

**Slide 18** — Analisis mendalam: Mengapa ResNet tidak degradasi dengan depth → skip connection. Kapan fine-tuning vs feature extraction? Kenapa Focal Loss untuk imbalance detection? Trade-off akurasi vs FPS di YOLO vs Faster R-CNN.

**Slide 19** — Koneksi antar modul: Feature extraction CNN (Modul 5) → embedding recognition (Modul 6). YOLO detection → tracking (Modul 9). Segmentasi → depth estimation (Modul 11). Transfer learning → fine-tune domain spesifik (seluruh CV pipeline).

**Slide 20** — Best practices: selalu normalize input [0,1]. Weight initialization (Xavier/He). Early stopping + learning rate scheduler. Mixed precision training (fp16). Gradient clipping. Checkpoint saving. Reproducibility: set seed.

**Slide 21** — Deployment pipeline: Train (GPU) → Evaluate → Export ONNX → Quantize (INT8) → Runtime (CPU/Edge). Perbandingan latency: PyTorch full vs ONNX vs TFLite vs TensorRT. Aplikasi: mobile, Raspberry Pi, Jetson Nano.

**Slide 22** — Perbandingan arsitektur: tabel LeNet/AlexNet/VGG/ResNet/MobileNet — parameter count, FLOPS, ImageNet top-1, inference time CPU. Kapan pilih MobileNet (mobile), kapan EfficientNet (akurasi tertinggi), kapan ResNet (baseline solid).

**Slide 23** — Checklist kompetensi: OpenCV DNN inference, CNN dari scratch, transfer learning, augmentasi, deteksi (HOG+YOLO), segmentasi, evaluasi mAP/IoU, ONNX export. Tabel self-assessment per percobaan.

**Slide 24** — Kuis: (1) Perbedaan semantic vs instance segmentation? (2) Mengapa BatchNorm mempercepat training? (3) IoU threshold YOLO default? (4) Apa yang di-freeze pada feature extraction? (5) Fungsi blobFromImage() di OpenCV DNN?

**Slide 25** — Diskusi: Overfitting pada dataset kecil → strategi augmentasi + transfer learning. mAP vs IoU: mana lebih informatif? Kapan ONNX lebih baik dari PyTorch? Trade-off Dice loss vs Cross-entropy untuk segmentasi medis.

**Slide 26** — Project "Sistem Deep Learning Computer Vision Terpadu". 10 soal cerita: Sortir Sampah (YOLO+klasifikasi), Driver Assistance (deteksi+segmentasi), Monitoring Kehadiran (face+counting), Identifikasi Tanaman (transfer learning), Quality Control PCB (defect detection), Monitoring Hewan (YOLO custom), Pembaca Buku Tunanetra (text detection+OCR), Analisis Citra Satelit (segmentasi), Fitness Counter (pose+counting), Fashion Classifier (klasifikasi+rekomendasi). Deliverable: .py, output/, laporan PDF.

**Slide 27** — 15 improvisasi: Custom YOLOv8, Real-time Face Mask Detector, Plant Disease Classifier, Multi-model Ensemble, GradCAM Visualizer, Augmentation Policy Searcher, Model Pruning Benchmark, Edge Deployment (Raspberry Pi/Jetson), Semi-supervised Learning, Semantic Segmentation Fine-tuner, Video Object Detection Pipeline, Mobile-optimized Classifier, Few-shot Learning, Auto-annotation Tool, Model Explainability Dashboard.

**Slide 28** — Rubrik Project: Fungsionalitas 35%, Integrasi Percobaan 20%, Kualitas Kode 15%, Dokumentasi 15%, Kreativitas 15%. Bonus: +5 hardware deployment, +5 real dataset custom. Deadline: 1 minggu setelah modul. Format: ZIP NIM_Nama_Project05.zip.

**Slide 29** — Tugas Video: 10–20 menit, screen+face-cam. Struktur: Pembukaan (2 mnt) — NIM/nama/kelas. Penjelasan Materi (10–15 mnt) — arsitektur CNN, transfer learning, YOLO, segmentasi, deployment. Demo 20 Percobaan LIVE (40+ mnt) — 2 poin/percobaan. Demo Project (10+ mnt). Analisis & Penutup. Submit YouTube/Drive → LMS.

**Slide 30** — Rubrik Video: Pembukaan (5), Materi (10), 20 Percobaan (40 — 2/percobaan), Project (30), Penutup (10), Kualitas (5). Total 100. Bonus +5 demo hardware edge, +3 GradCAM real-time. Penalti −2/percobaan tidak tampil. "Kuasai Deep Learning, Bangun Sistem Cerdas nyata!"
