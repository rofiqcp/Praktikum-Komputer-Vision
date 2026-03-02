# NotebookLM Prompts — Modul 9: Deep Learning untuk Komputer Vision

---

## PROMPT 1 — Slide 1–15 (Materi + Jobsheet Bagian 1)

Buat 15 slide presentasi akademik Modul 9: Deep Learning untuk Komputer Vision. Referensi Szeliski (2022) Ch.5. Tiap slide ~500 kata, sertakan diagram dan kode Python.

**Slide 1** — Judul "Modul 9: Deep Learning untuk Komputer Vision", subtitle "Dari Piksel ke Prediksi dengan Jaringan Saraf Dalam", ilustrasi CNN pipeline input→conv→pool→FC→output, referensi Szeliski Ch.5.

**Slide 2** — Sejarah DL: Perceptron (1958) → MLP → LeNet (1998) → AlexNet (2012, ImageNet breakthrough) → VGG/GoogLeNet/ResNet (2014-15) → MobileNet (2017) → EfficientNet (2019) → ViT (2020). Diagram timeline revolusi deep learning.

**Slide 3** — ANN: Neuron y=σ(wᵀx+b). Aktivasi: ReLU=max(0,z), Sigmoid=1/(1+e⁻ᶻ), Tanh, LeakyReLU=max(αz,z), Softmax. Training: forward pass → loss → backpropagation → gradient descent. Optimizer: SGD, Momentum, Adam.

**Slide 4** — CNN arsitektur: filter shared spasial, y_ij=ΣΣw_kl·x_{i+k,j+l}+b. Komponen: Conv→ReLU→Pool→BN→Dropout→FC. Parameter: kernel size, stride, padding. MaxPool vs AvgPool. Diagram feature map progression.

**Slide 5** — Arsitektur terkenal: LeNet-5 (60K param), AlexNet (60M, GPU+ReLU), VGG (138M, 3×3 uniform), GoogLeNet/Inception (5M, multi-scale), ResNet (25M, skip connection), MobileNet (3.4M, depthwise separable), EfficientNet (5.3M, compound scaling). Tabel parameter+akurasi.

**Slide 6** — Transfer Learning: 3 strategi — (1) From Scratch: semua layer dari awal, butuh data besar. (2) Feature Extraction: freeze semua, train FC saja. (3) Fine-tuning: unfreeze layer terakhir, lr kecil. Diagram ImageNet→domain spesifik. Percobaan 8.

**Slide 7** — Data Augmentation: Geometric (flip, rotation ±15°, crop, scale). Photometric (brightness, contrast, noise Gaussian). Advanced: Cutout, Mixup, CutMix. Tujuan: cegah overfitting, tingkatkan generalisasi. Grid 1 gambar × 8 augmentasi. Percobaan 7.

**Slide 8** — Object Detection: Two-stage (R-CNN→Faster R-CNN: region proposal+classify). One-stage (YOLO: grid S×S, prediksi B bbox+C class per cell. SSD, RetinaNet). Sliding window + image pyramid (klasik). NMS post-processing. Percobaan 10,12.

**Slide 9** — HOG Descriptor: (1) gradient magnitude+orientasi, (2) cell histogram 8×8, (3) block normalization 2×2, (4) gabung → vektor fitur → SVM. cv2.HOGDescriptor pedestrian detection. HOG vs DL: trade-off kecepatan vs akurasi. Percobaan 11.

**Slide 10** — Segmentation: Semantic=klasifikasi per-piksel (FCN, UNet, DeepLab). Instance=deteksi+mask per objek (Mask R-CNN). Manual: HSV color-based, K-Means clustering, thresholding+morphology. Dice coefficient, mIoU. Percobaan 13-14.

**Slide 11** — Percobaan 1-2: OpenCV DNN preprocessing. cv2.dnn.blobFromImage(scalefactor, size, mean, swapRB). Blob shape NCHW. Klasifikasi DNN: readNet→setInput→forward→top-5. Waktu inferensi. Simulasi jika model tidak tersedia.

**Slide 12** — Percobaan 3-5: Perbandingan 5 model (radar chart akurasi/param/speed/size). Visualisasi arsitektur CNN layer-by-layer (conv→ReLU→pool output). Fungsi aktivasi: plot ReLU/Sigmoid/Tanh/LeakyReLU + turunannya + efek pada gambar.

**Slide 13** — Percobaan 6-9: Konvolusi manual (Sobel, Gaussian, Sharpen kernel) + pooling (MaxPool vs AvgPool). Data augmentasi 8 teknik. Transfer learning 3 strategi diagram + training curves. Backpropagation XOR: loss curve, gradient, decision boundary.

**Slide 14** — Percobaan 10-14: Sliding window + pyramid + NMS. HOG descriptor + pedestrian detection. YOLO grid overlay + IoU + anchor boxes. Semantic segmentation manual (HSV + K-Means + overlay). Instance segmentation: per-instance mask + 4-task comparison diagram.

**Slide 15** — Percobaan 15-20: Loss functions plot (MSE/BCE/CCE/Dice/Focal + 3D landscape). Optimizer trajectory (SGD/Momentum/Adam/RMSProp contour). BatchNorm distribusi + Dropout efek. Metrik: confusion matrix + ROC + PR curve. ONNX pipeline + quantization. Proyek klasifikasi bentuk end-to-end.

---

## PROMPT 2 — Slide 16–30 (Materi Lanjutan + Analisis)

Lanjutkan slide presentasi Modul 9: Deep Learning untuk Komputer Vision, Slide 16–30. Tiap slide ~500 kata. Fokus analisis mendalam, koneksi antar konsep, best practices, dan rekap.

**Slide 16** — Loss Functions mendalam: MSE=Σ(y-ŷ)²/n (regresi), BCE=-[y·log(ŷ)+(1-y)·log(1-ŷ)] (binary), CCE=-Σyᵢlog(ŷᵢ) (multi-class), Dice=1-2|A∩B|/(|A|+|B|) (segmentasi), Focal=-αₜ(1-pₜ)ᵧlog(pₜ) (imbalance). Kapan pakai apa. Tabel perbandingan.

**Slide 17** — Optimizer mendalam: SGD θ=θ-η∇L. Momentum v=βv+η∇L, θ=θ-v. RMSProp s=βs+(1-β)g², adaptive lr. Adam=Momentum+RMSProp, m dan v estimasi + bias correction. Trajectory pada Rosenbrock function. Konvergensi comparison.

**Slide 18** — Batch Normalization detail: normalize per mini-batch μ_B, σ_B. Learnable γ, β. Training vs inference (running mean). Keuntungan: stabilisasi, lr besar, mengurangi internal covariate shift. Dropout detail: rate 0.2-0.5, inverted dropout, regularisasi. BN+Dropout kombinasi.

**Slide 19** — Metrik evaluasi detail: Confusion Matrix TP/TN/FP/FN. Accuracy (misleading pada imbalanced data). Precision→kapan FP mahal. Recall→kapan FN mahal. F1 harmonic mean. ROC: TPR vs FPR, AUC ideal=1. PR Curve: fokus kelas positif. mAP@50, mAP@75.

**Slide 20** — ONNX & Deployment: Format universal model DNN. Pipeline: Train→Export→Optimize→Deploy. Quantization FP32→INT8 (4x kompresi, sedikit akurasi turun). OpenCV DNN: readNet tanpa framework berat. Runtime comparison: ONNX Runtime, TensorRT, OpenVINO, TFLite.

**Slide 21** — Rekap Percobaan 1-10: (1) DNN blob preprocessing, (2) DNN klasifikasi, (3) model comparison radar, (4) CNN arsitektur visualisasi, (5) fungsi aktivasi, (6) konvolusi+pooling, (7) augmentasi, (8) transfer learning, (9) backpropagation, (10) sliding window. Tabel file+konsep+output.

**Slide 22** — Rekap Percobaan 11-20: (11) HOG detection, (12) YOLO grid, (13) semantic segmentation, (14) instance segmentation, (15) loss functions, (16) optimizer, (17) BN+Dropout, (18) evaluasi metrik, (19) ONNX deployment, (20) proyek bentuk. Tabel file+library+output.

**Slide 23** — Analisis mendalam: Mengapa ResNet tidak degradasi → skip connection gradient highway. Feature extraction vs fine-tuning: kapan dan mengapa. Focal Loss γ parameter mengatasi easy example dominasi. HOG+SVM vs CNN: accuracy vs speed vs data requirement.

**Slide 24** — Koneksi antar modul: CNN features (Modul 9) → recognition embedding (Modul 10). YOLO detection → tracking (Modul 7). Segmentasi → depth estimation (Modul 11). Transfer learning → fine-tune semua domain. Feature extraction → 3D reconstruction (Modul 12).

**Slide 25** — Best practices DL: (1) Normalize input [0,1] atau ImageNet mean/std. (2) Weight init Xavier/He. (3) Start simple, add complexity. (4) Early stopping + LR scheduler. (5) BN sebelum activation. (6) Dropout setelah FC. (7) Adam sebagai default optimizer. (8) Monitor val loss, bukan train loss.

**Slide 26** — Trade-off arsitektur: tabel LeNet/AlexNet/VGG/ResNet/MobileNet/EfficientNet — params, FLOPs, top-1 ImageNet, CPU inference ms. Kapan MobileNet (mobile/edge), EfficientNet (akurasi max), ResNet (baseline solid, banyak pretrained).

**Slide 27** — Perbandingan deteksi: Sliding window (exhaustive, lambat, no learning) vs HOG+SVM (fitur manual, cepat CPU, rigid objects) vs YOLO (end-to-end, cepat GPU, flexible). Tabel speed/accuracy/training data/hardware. Evolusi: hand-crafted → learned features.

**Slide 28** — Perbandingan segmentasi: Manual (HSV threshold: sederhana, cepat, domain-specific) vs K-Means (unsupervised, no labels needed) vs FCN/UNet (learned, akurat, butuh data+GPU). Semantic vs Instance: kapan mana yang dipakai. mIoU benchmark.

**Slide 29** — Checklist kompetensi: ☐ blob preprocessing, ☐ DNN inference, ☐ CNN arsitektur, ☐ aktivasi, ☐ konvolusi manual, ☐ augmentasi, ☐ transfer learning, ☐ backprop, ☐ sliding window, ☐ HOG, ☐ YOLO grid, ☐ segmentasi, ☐ loss function, ☐ optimizer, ☐ BN/Dropout, ☐ metrik evaluasi, ☐ ONNX. Self-assessment per percobaan.

**Slide 30** — Kuis: (1) Apa fungsi blobFromImage? (2) Perbedaan MaxPool vs AvgPool? (3) Mengapa ReLU lebih baik dari Sigmoid? (4) Apa itu skip connection ResNet? (5) IoU threshold default YOLO? (6) Kapan Dice Loss lebih baik dari CE? (7) Adam vs SGD? (8) Apa inverted dropout?

---

## PROMPT 3 — Slide 31–45 (Materi Lanjutan + Project + Tugas Video)

Lanjutkan slide Modul 9, Slide 31–45. Slide 31–35: diskusi dan topik lanjutan. Slide 36–41: Project. Slide 42–45: Tugas Video. Tiap slide ~500 kata.

**Slide 31** — Diskusi: Overfitting pada dataset kecil → augmentasi + transfer learning + dropout + early stopping. Class imbalance → Focal Loss + oversampling + weighted loss. Generalisasi → regularisasi + diverse data + ensemble. Real-world vs benchmark performance.

**Slide 32** — Topik lanjutan 1: Grad-CAM (Gradient-weighted Class Activation Mapping) — visualisasi apa yang "dilihat" CNN. Menghitung gradient output terhadap feature map → weighted sum → heatmap overlay. Penting untuk interpretability dan debugging model.

**Slide 33** — Topik lanjutan 2: Autoencoder (encoder→bottleneck→decoder) untuk feature learning unsupervised. Variational Autoencoder (VAE). GAN (Generator vs Discriminator) untuk data synthesis. Vision Transformer (ViT): patch embedding + self-attention menggantikan konvolusi.

**Slide 34** — Topik lanjutan 3: Callbacks training best practices: ModelCheckpoint (save best), EarlyStopping (prevent overfit), ReduceLROnPlateau (adaptive lr). TensorBoard monitoring: loss, accuracy, histograms, images. tf.data pipeline untuk efficient loading.

**Slide 35** — Pipeline CV lengkap: Data Collection → Augmentation → Model Selection (CNN/pretrained) → Training (loss+optimizer+BN+Dropout) → Evaluation (metrics) → Export (ONNX) → Deployment (quantize) → Monitoring → Iterate. Diagram alur end-to-end.

**Slide 36** — Project overview: "Sistem Deep Learning CV Terpadu". Integrasikan ≥5 percobaan. 10 soal cerita: (1) Sortir Sampah, (2) Driver Assistance, (3) Monitoring Kehadiran, (4) Identifikasi Tanaman, (5) Quality Control PCB, (6) Monitoring Hewan, (7) Pembaca Buku Tunanetra, (8) Citra Satelit, (9) Fitness Counter, (10) Fashion Classifier.

**Slide 37** — Soal cerita detail 1-5: Sortir Sampah (YOLO+classifier+webcam+statistik+ONNX). Driver Assistance (deteksi+segmentasi+jarak+alert). Monitoring Kehadiran (deteksi+counting+instance seg+grafik). Identifikasi Tanaman (transfer learning+GradCAM+top-3). Quality Control (defect detection+segmentasi+heatmap+pass/fail).

**Slide 38** — Soal cerita detail 6-10: Monitoring Hewan (YOLO custom+tracking+aktivitas). Pembaca Buku (text detection+segmentasi+OCR+batch). Citra Satelit (semantic seg+persentase+temporal+change map). Fitness Counter (deteksi+pose+counting+statistik). Fashion Classifier (klasifikasi+segmentasi+warna+rekomendasi).

**Slide 39** — 15 Improvisasi: Custom YOLOv8, Face Mask Detector, Plant Disease, Multi-model Ensemble, GradCAM Visualizer, Augmentation Searcher, Model Pruning, Edge Deploy (RPi/Jetson), Semi-supervised, Semantic Seg Fine-tuner, Video Detection Pipeline, Mobile Classifier, Few-shot, Auto-annotation, Explainability Dashboard.

**Slide 40** — Rubrik Project: Fungsionalitas 35% (semua fitur berjalan, output benar). Integrasi 20% (≥5 percobaan digunakan). Kode 15% (clean, documented, def/function style). Dokumentasi 15% (laporan PDF lengkap). Kreativitas 15% (inovasi, UI, performa). Format: ZIP NIM_Nama_Project09.zip.

**Slide 41** — Contoh implementasi project: Pipeline sortir sampah — (1) download dataset 5 kelas, (2) augmentasi, (3) train classifier transfer learning, (4) YOLO deteksi objek, (5) OpenCV webcam realtime, (6) statistik matplotlib, (7) export ONNX. Kode snippet tiap tahap.

**Slide 42** — Tugas Video overview: 75-100 menit. Struktur: Pembukaan (2 mnt, NIM+nama+kelas) → Materi (10-15 mnt, CNN+TL+YOLO+segmentasi+deployment) → Demo 20 Percobaan LIVE (40-60 mnt) → Demo Project (10-15 mnt) → Analisis & Penutup (5 mnt).

**Slide 43** — Demo percobaan detail: Percobaan 1-10 (blob, DNN klasifikasi, model comparison, CNN arsitektur, aktivasi, konvolusi+pooling, augmentasi, transfer learning, backprop, sliding window). Percobaan 11-20 (HOG, YOLO grid, semantic seg, instance seg, loss, optimizer, BN+Dropout, metrik, ONNX, proyek bentuk).

**Slide 44** — Rubrik Video: Pembukaan 5%, Materi 15%, Demo 20 Percobaan 40% (2%/percobaan), Project 20%, Analisis 10%, Kualitas 10%. Bonus: +5 GPU demo, +5 perbandingan kuantitatif detail. Penalti: -10 video<60mnt, -5 no webcam, -5 audio buruk, -5/percobaan error.

**Slide 45** — Penutup dan call-to-action: Deep learning adalah fondasi CV modern. Kuasai: preprocessing→arsitektur→training→evaluasi→deployment. Dari Modul 9 ke Modul 10 (Recognition): CNN features → face/object recognition. Submit: YouTube/Drive link + ZIP project. "Bangun sistem cerdas dari piksel ke prediksi!"
