# JOBSHEET PRAKTIKUM
# MODUL 5: DEEP LEARNING UNTUK KOMPUTER VISION

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Melakukan klasifikasi gambar menggunakan model pre-trained via OpenCV DNN.
2. Membandingkan performa beberapa model klasifikasi.
3. Membangun dan melatih CNN sederhana dengan PyTorch.
4. Membangun dan melatih CNN sederhana dengan Keras/TensorFlow.
5. Menerapkan transfer learning untuk klasifikasi custom dataset.
6. Mengimplementasikan data augmentation dan menganalisis pengaruhnya.
7. Melakukan deteksi objek menggunakan YOLOv5/v8.
8. Menjalankan deteksi objek real-time dari webcam.
9. Melakukan semantic segmentation menggunakan model pre-trained.
10. Melakukan instance segmentation dan export model ke ONNX.
11. Mengimplementasikan deteksi objek menggunakan HOG (Histogram of Oriented Gradients).
12. Memahami konsep grid dan prediksi pada arsitektur YOLO.
13. Mengimplementasikan semantic segmentation secara manual.
14. Memahami konsep instance segmentation dan perbedaannya dengan semantic segmentation.
15. Memvisualisasikan dan memahami berbagai loss function (Cross-Entropy, MSE, Dice).
16. Membandingkan perilaku berbagai optimizer (SGD, Adam, SGDM).
17. Memahami dan mengimplementasikan Batch Normalization dan Dropout.
18. Menghitung dan menganalisis metrik evaluasi model (accuracy, precision, recall, F1, mAP, IoU).
19. Memahami konsep ONNX export dan deployment model.
20. Membuat proyek klasifikasi bentuk end-to-end sebagai integrasi seluruh konsep.

---

## 2. ALAT DAN BAHAN

### Perangkat Keras
- Laptop/PC (GPU NVIDIA direkomendasikan, CPU juga bisa)
- Webcam

### Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `numpy`, `matplotlib`, `torch`, `torchvision`, `tensorflow`, `keras`, `ultralytics`, `onnx`, `onnxruntime`

### Dataset
- CIFAR-10 atau MNIST (auto-download via torchvision/keras).
- Custom dataset kecil (5 kelas, 50 gambar per kelas).
- Pre-trained model weights (auto-download).

---

## 3. LANGKAH KERJA

### Percobaan 1: Klasifikasi Gambar dengan OpenCV DNN

**Tujuan**: Menggunakan model pre-trained untuk klasifikasi gambar tanpa framework DL.

**Langkah Kerja**:
1. Buat file `01_opencv_dnn_classification.py`.
2. Download model pre-trained (MobileNet, GoogLeNet) dalam format Caffe/ONNX.
3. Load model: `net = cv2.dnn.readNet(model, config)`.
4. Buat blob dari gambar: `blob = cv2.dnn.blobFromImage(img, scalefactor, size, mean)`.
5. Inferensi: `net.setInput(blob)` → `output = net.forward()`.
6. Parse output: temukan class dengan probabilitas tertinggi.
7. Load label ImageNet (1000 kelas).
8. Klasifikasi 5 gambar berbeda dan tampilkan top-5 prediksi.
9. Ukur waktu inferensi per gambar.
10. Tampilkan hasilnya dengan gambar + label + confidence.

---

### Percobaan 2: Perbandingan Model Pre-trained

**Tujuan**: Membandingkan akurasi dan kecepatan beberapa model klasifikasi.

**Langkah Kerja**:
1. Buat file `02_model_comparison.py`.
2. Siapkan minimal 3 model: MobileNetV2, ResNet50, GoogLeNet.
3. Klasifikasi 10 gambar yang sama dengan ketiga model.
4. Catat: top-1 accuracy, top-5 accuracy, waktu inferensi, ukuran model.
5. Buat tabel perbandingan.
6. Buat grafik: akurasi vs kecepatan vs ukuran.
7. Analisis trade-off dari setiap model.
8. Identifikasi gambar yang salah diklasifikasi dan analisis penyebabnya.
9. Coba variasikan input size.
10. Simpan laporan perbandingan.

---

### Percobaan 3: CNN dengan PyTorch

**Tujuan**: Membangun, melatih, dan mengevaluasi CNN dari scratch menggunakan PyTorch.

**Langkah Kerja**:
1. Buat file `03_cnn_pytorch.py`.
2. Load dataset CIFAR-10 menggunakan `torchvision.datasets`.
3. Definisikan arsitektur CNN:
   - 3 Convolutional layers + ReLU + MaxPool.
   - 2 Fully Connected layers.
   - Output: 10 kelas.
4. Definisikan loss: `nn.CrossEntropyLoss()`.
5. Definisikan optimizer: `optim.Adam(lr=0.001)`.
6. Training loop: 10–20 epoch, print loss per epoch.
7. Evaluasi pada test set: hitung akurasi.
8. Plot training loss curve.
9. Tampilkan confusion matrix.
10. Simpan model: `torch.save()`.

---

### Percobaan 4: CNN dengan Keras/TensorFlow

**Tujuan**: Membangun CNN menggunakan Keras dan membandingkan dengan PyTorch.

**Langkah Kerja**:
1. Buat file `04_cnn_keras.py`.
2. Load dataset CIFAR-10 dengan `keras.datasets`.
3. Bangun model Sequential:
   - `Conv2D → ReLU → MaxPooling2D` × 3.
   - `Flatten → Dense → Dropout → Dense(10, softmax)`.
4. Compile: `model.compile(optimizer='adam', loss='categorical_crossentropy')`.
5. Train: `model.fit(X_train, y_train, epochs=10, validation_split=0.2)`.
6. Evaluasi: `model.evaluate(X_test, y_test)`.
7. Plot training/validation accuracy dan loss.
8. Bandingkan akurasi dan waktu training dengan PyTorch (Percobaan 3).
9. Visualisasikan filter layer pertama.
10. Simpan model: `model.save()`.

---

### Percobaan 5: Transfer Learning

**Tujuan**: Fine-tune model pre-trained untuk klasifikasi dataset custom.

**Langkah Kerja**:
1. Buat file `05_transfer_learning.py`.
2. Siapkan dataset custom: 5 kelas, 50–100 gambar per kelas.
3. Load model ResNet18 pre-trained (PyTorch) atau MobileNetV2 (Keras).
4. Freeze semua layer kecuali classifier terakhir.
5. Ganti classifier: `model.fc = nn.Linear(512, 5)`.
6. Train hanya classifier baru: 5–10 epoch.
7. Unfreeze 2–3 layer terakhir dan fine-tune: 5 epoch lagi.
8. Evaluasi: akurasi, confusion matrix.
9. Bandingkan: transfer learning vs training from scratch.
10. Visualisasikan prediksi (gambar + label + confidence).

---

### Percobaan 6: Data Augmentation

**Tujuan**: Mengimplementasikan augmentasi data dan menganalisis dampaknya terhadap performa.

**Langkah Kerja**:
1. Buat file `06_data_augmentation.py`.
2. Siapkan dataset kecil (untuk melihat efek augmentasi jelas).
3. Implementasikan augmentasi:
   - Random horizontal flip, rotation (±15°), crop.
   - Color jitter (brightness, contrast, saturation).
   - Random erasing / Cutout.
4. Visualisasikan 5 versi augmented dari 1 gambar yang sama.
5. Train model **tanpa** augmentasi → catat akurasi.
6. Train model **dengan** augmentasi → catat akurasi.
7. Bandingkan: akurasi pada validation set + training curve.
8. Coba augmentasi agresif vs konservatif.
9. Implementasikan `albumentations` library untuk augmentasi lebih lanjut.
10. Buat laporan dampak augmentasi.

---

### Percobaan 7: Deteksi Objek YOLO

**Tujuan**: Menggunakan YOLO untuk mendeteksi objek pada gambar.

**Langkah Kerja**:
1. Buat file `07_yolo_detection.py`.
2. Install `ultralytics`: `pip install ultralytics`.
3. Load model: `model = YOLO('yolov8n.pt')` (nano model).
4. Deteksi pada gambar: `results = model(img)`.
5. Tampilkan bounding box, label, dan confidence score.
6. Terapkan pada 5 gambar berbeda (indoor, outdoor, crowd, traffic).
7. Coba model berbeda: yolov8n, yolov8s, yolov8m.
8. Filter deteksi berdasarkan kelas tertentu (person, car, dll.).
9. Filter berdasarkan confidence threshold.
10. Simpan gambar hasil deteksi.

---

### Percobaan 8: Deteksi Objek Real-time

**Tujuan**: Menjalankan deteksi objek secara real-time dari webcam.

**Langkah Kerja**:
1. Buat file `08_yolo_realtime.py`.
2. Buka webcam: `cap = cv2.VideoCapture(0)`.
3. Loop: baca frame → deteksi YOLO → tampilkan dengan anotasi.
4. Tampilkan FPS di layar.
5. Hitung jumlah objek terdeteksi per frame.
6. Implementasikan toggle ON/OFF deteksi (tekan tombol).
7. Record video output dengan deteksi.
8. Bandingkan FPS: yolov8n vs yolov8s.
9. Tampilkan statistik: jumlah deteksi per kelas.
10. Graceful exit: simpan video dan statistik.

---

### Percobaan 9: Semantic Segmentation

**Tujuan**: Melakukan segmentasi semantik per-piksel menggunakan model pre-trained.

**Langkah Kerja**:
1. Buat file `09_semantic_segmentation.py`.
2. Load model segmentation (DeepLabV3 via torchvision atau OpenCV DNN).
3. Preprocessing gambar sesuai requirement model.
4. Inferensi: dapatkan prediction mask.
5. Color-code setiap kelas dengan warna berbeda.
6. Overlay segmentation mask pada gambar asli (semi-transparan).
7. Terapkan pada 5 gambar berbeda.
8. Hitung persentase area setiap kelas.
9. Visualisasikan per-class mask secara terpisah.
10. Simpan visualisasi.

---

### Percobaan 10: Instance Segmentation dan ONNX Export

**Tujuan**: Melakukan instance segmentation dan mengeksport model ke format ONNX.

**Langkah Kerja**:
1. Buat file `10_instance_segmentation.py`.
2. Load model instance segmentation (YOLO-seg atau Mask R-CNN).
3. Deteksi dan segmentasi setiap instance objek.
4. Tampilkan: bounding box + mask + label per instance.
5. Beri warna berbeda untuk setiap instance.
6. Terapkan pada 3 gambar dengan banyak objek.
7. Export model ke ONNX: `model.export(format='onnx')` atau `torch.onnx.export()`.
8. Load model ONNX dengan OpenCV: `cv2.dnn.readNetFromONNX()`.
9. Bandingkan inferensi: framework asli vs ONNX.
10. Catat ukuran model, waktu inferensi, dan akurasi.

---

### Percobaan 11: Deteksi Objek dengan HOG

**Tujuan**: Mengimplementasikan deteksi objek menggunakan fitur HOG (Histogram of Oriented Gradients).

**Langkah Kerja**:
1. Buat file `11_deteksi_objek_hog.py`.
2. Load gambar dan konversi ke grayscale.
3. Hitung HOG descriptor menggunakan `cv2.HOGDescriptor()`.
4. Gunakan detektor bawaan: `hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())`.
5. Deteksi orang pada gambar: `hog.detectMultiScale()`.
6. Gambar bounding box pada hasil deteksi.
7. Visualisasikan fitur HOG (orientasi gradien) pada gambar.
8. Variasikan parameter: `winStride`, `padding`, `scale`.
9. Terapkan pada minimal 3 gambar berbeda.
10. Analisis kelebihan dan kekurangan HOG dibanding deep learning.

---

### Percobaan 12: YOLO Konsep Grid

**Tujuan**: Memahami konsep pembagian grid dan prediksi pada arsitektur YOLO.

**Langkah Kerja**:
1. Buat file `12_yolo_konsep_grid.py`.
2. Load gambar dan tentukan ukuran grid (misalnya 7×7 atau 13×13).
3. Gambar grid overlay pada gambar.
4. Simulasikan proses prediksi: setiap cell memprediksi bounding box.
5. Visualisasikan anchor boxes pada beberapa cell.
6. Implementasikan perhitungan IoU antara predicted box dan ground truth.
7. Terapkan Non-Maximum Suppression (NMS) secara manual.
8. Visualisasikan hasil sebelum dan sesudah NMS.
9. Variasikan jumlah grid dan amati pengaruhnya.
10. Buat diagram alur pipeline deteksi YOLO.

---

### Percobaan 13: Semantic Segmentation Manual

**Tujuan**: Mengimplementasikan konsep semantic segmentation secara manual tanpa model pre-trained.

**Langkah Kerja**:
1. Buat file `13_semantic_segmentation_manual.py`.
2. Load gambar dengan objek yang mudah di-segmentasi.
3. Terapkan thresholding untuk segmentasi dasar.
4. Gunakan color-based segmentation (HSV range).
5. Terapkan operasi morfologi untuk memperbaiki mask.
6. Beri label kelas pada setiap region yang tersegmentasi.
7. Buat color-coded segmentation map.
8. Overlay segmentation map pada gambar asli (semi-transparan).
9. Hitung persentase area setiap kelas.
10. Bandingkan hasil segmentasi manual vs model pre-trained.

---

### Percobaan 14: Instance Segmentation (Konsep)

**Tujuan**: Memahami perbedaan dan konsep instance segmentation dibanding semantic segmentation.

**Langkah Kerja**:
1. Buat file `14_instance_segmentation_konsep.py`.
2. Load gambar dengan beberapa objek dari kelas yang sama.
3. Terapkan segmentasi per-instance menggunakan connected components.
4. Beri warna unik pada setiap instance yang terdeteksi.
5. Hitung jumlah instance per kelas.
6. Visualisasikan bounding box + mask per instance.
7. Implementasikan pemisahan instance yang berdekatan (watershed).
8. Bandingkan hasil: semantic segmentation vs instance segmentation.
9. Tampilkan statistik: jumlah instance, area rata-rata, overlap.
10. Buat visualisasi side-by-side semantic vs instance.

---

### Percobaan 15: Loss Function Visualisasi

**Tujuan**: Memvisualisasikan dan memahami berbagai loss function yang digunakan dalam deep learning.

**Langkah Kerja**:
1. Buat file `15_loss_function_visualisasi.py`.
2. Implementasikan Mean Squared Error (MSE) loss secara manual.
3. Implementasikan Cross-Entropy loss secara manual.
4. Implementasikan Dice loss secara manual.
5. Buat plot loss vs predicted value untuk setiap loss function.
6. Visualisasikan loss landscape dalam 2D/3D.
7. Bandingkan perilaku loss untuk klasifikasi biner.
8. Analisis efek class imbalance pada setiap loss function.
9. Implementasikan Focal Loss dan bandingkan dengan Cross-Entropy.
10. Buat ringkasan kapan menggunakan loss function yang mana.

---

### Percobaan 16: Optimizer Visualisasi

**Tujuan**: Membandingkan perilaku berbagai optimizer dalam proses training.

**Langkah Kerja**:
1. Buat file `16_optimizer_visualisasi.py`.
2. Buat fungsi loss sederhana (Rosenbrock, Rastrigin, atau quadratic).
3. Implementasikan SGD (Stochastic Gradient Descent) secara manual.
4. Implementasikan SGD with Momentum (SGDM) secara manual.
5. Implementasikan Adam optimizer secara manual.
6. Visualisasikan trajectory optimasi pada contour plot.
7. Bandingkan kecepatan konvergensi ketiga optimizer.
8. Variasikan learning rate dan amati pengaruhnya.
9. Plot loss curve per iterasi untuk setiap optimizer.
10. Analisis kapan menggunakan optimizer yang mana.

---

### Percobaan 17: Batch Normalization dan Dropout

**Tujuan**: Memahami dan memvisualisasikan efek Batch Normalization dan Dropout pada training.

**Langkah Kerja**:
1. Buat file `17_batch_normalization_dropout.py`.
2. Buat dataset sintetis atau load dataset sederhana.
3. Visualisasikan distribusi aktivasi tanpa Batch Normalization.
4. Terapkan Batch Normalization dan visualisasikan distribusi setelahnya.
5. Implementasikan Dropout secara manual (masking random neurons).
6. Visualisasikan efek Dropout pada arsitektur jaringan.
7. Bandingkan training: tanpa BN/Dropout vs dengan BN vs dengan Dropout vs keduanya.
8. Plot training dan validation loss untuk setiap konfigurasi.
9. Analisis efek regularisasi Dropout terhadap overfitting.
10. Buat ringkasan best practices penggunaan BN dan Dropout.

---

### Percobaan 18: Model Evaluasi Metrik

**Tujuan**: Menghitung dan memvisualisasikan berbagai metrik evaluasi model deep learning.

**Langkah Kerja**:
1. Buat file `18_model_evaluasi_metrik.py`.
2. Buat data prediksi dan ground truth simulasi.
3. Hitung Accuracy, Precision, Recall, dan F1-Score secara manual.
4. Buat confusion matrix dan visualisasikan sebagai heatmap.
5. Hitung IoU (Intersection over Union) untuk segmentasi.
6. Hitung mAP (mean Average Precision) untuk deteksi objek.
7. Plot Precision-Recall curve.
8. Plot ROC curve dan hitung AUC.
9. Analisis metrik pada dataset yang imbalanced.
10. Buat dashboard ringkasan semua metrik dalam satu visualisasi.

---

### Percobaan 19: ONNX dan Deployment

**Tujuan**: Memahami konsep export model ke ONNX dan proses deployment.

**Langkah Kerja**:
1. Buat file `19_onnx_dan_deployment_konsep.py`.
2. Buat model sederhana (PyTorch atau simulasi).
3. Export model ke format ONNX.
4. Visualisasikan graph model ONNX.
5. Load model ONNX menggunakan ONNX Runtime.
6. Bandingkan inferensi: model asli vs ONNX Runtime.
7. Ukur waktu inferensi dan penggunaan memori.
8. Simulasikan quantization (FP32 → FP16).
9. Buat diagram pipeline deployment model.
10. Analisis trade-off akurasi vs kecepatan setelah optimasi.

---

### Percobaan 20: Proyek Klasifikasi Bentuk

**Tujuan**: Membuat proyek klasifikasi bentuk geometris end-to-end sebagai integrasi seluruh konsep.

**Langkah Kerja**:
1. Buat file `20_proyek_klasifikasi_bentuk.py`.
2. Generate dataset sintetis bentuk geometris (lingkaran, segitiga, kotak, bintang, dll.).
3. Terapkan data augmentasi pada dataset.
4. Bangun arsitektur CNN sederhana untuk klasifikasi.
5. Train model dengan loss function dan optimizer yang sesuai.
6. Terapkan Batch Normalization dan Dropout.
7. Evaluasi dengan semua metrik (accuracy, precision, recall, F1, confusion matrix).
8. Visualisasikan prediksi pada test set.
9. Export model ke ONNX dan verifikasi inferensi.
10. Buat laporan lengkap: arsitektur, training curve, metrik, dan analisis.

---

## 4. ANALISIS

### Percobaan 1 — OpenCV DNN
- Apa keuntungan menggunakan OpenCV DNN dibanding PyTorch/Keras untuk inferensi?
- Apakah akurasi berbeda saat menggunakan OpenCV vs framework asli?

### Percobaan 2 — Model Comparison
- Model mana yang optimal untuk embedded/mobile deployment?
- Trade-off apa yang paling penting untuk real-time application?

### Percobaan 3 & 4 — CNN PyTorch & Keras
- Bandingkan kemudahan coding PyTorch vs Keras.
- Apakah akurasi berbeda? Mengapa?

### Percobaan 5 — Transfer Learning
- Berapa besar improvement transfer learning vs from scratch?
- Berapa minimum data yang diperlukan untuk fine-tuning yang baik?

### Percobaan 6 — Data Augmentation
- Augmentasi mana yang paling berpengaruh?
- Apakah augmentasi terlalu agresif bisa menurunkan performa?

### Percobaan 7 & 8 — YOLO
- Apa trade-off antara model size (n, s, m, l)?
- Berapa FPS minimum yang acceptable untuk real-time?

### Percobaan 9 — Semantic Segmentation
- Kelas mana yang paling sering salah di-segmentasi?
- Bagaimana resolusi input mempengaruhi kualitas segmentasi?

### Percobaan 10 — Instance Segmentation & ONNX
- Apa perbedaan semantic vs instance segmentation?
- Apakah ONNX export mempengaruhi akurasi?

### Percobaan 11 — HOG Detection
- Bagaimana performa HOG dibandingkan deep learning-based detector?
- Pada kondisi apa HOG masih relevan digunakan?

### Percobaan 12 — YOLO Grid
- Bagaimana ukuran grid mempengaruhi kemampuan deteksi objek kecil vs besar?
- Mengapa NMS diperlukan dan bagaimana parameter threshold mempengaruhi hasil?

### Percobaan 13 — Semantic Segmentation Manual
- Apa keterbatasan segmentasi manual dibanding model deep learning?
- Pada skenario apa segmentasi berbasis warna masih efektif?

### Percobaan 14 — Instance Segmentation (Konsep)
- Apa tantangan utama dalam memisahkan instance yang saling berdekatan?
- Kapan instance segmentation lebih diperlukan dibanding semantic segmentation?

### Percobaan 15 — Loss Function
- Mengapa pemilihan loss function yang tepat sangat penting?
- Bagaimana Focal Loss mengatasi masalah class imbalance?

### Percobaan 16 — Optimizer
- Mengapa Adam lebih cepat konvergen dibanding SGD pada umumnya?
- Kapan SGD with momentum lebih baik dari Adam?

### Percobaan 17 — Batch Normalization & Dropout
- Bagaimana Batch Normalization mempercepat training?
- Apakah Dropout dan Batch Normalization selalu meningkatkan performa?

### Percobaan 18 — Evaluasi Metrik
- Mengapa accuracy saja tidak cukup untuk mengevaluasi model?
- Metrik mana yang paling penting untuk task deteksi vs klasifikasi?

### Percobaan 19 — ONNX & Deployment
- Apa keuntungan utama menggunakan ONNX untuk deployment?
- Bagaimana quantization mempengaruhi akurasi dan kecepatan?

### Percobaan 20 — Proyek Klasifikasi Bentuk
- Bagaimana integrasi seluruh konsep (augmentasi, BN, dropout, metrik) meningkatkan performa?
- Apa langkah-langkah penting untuk memastikan model siap di-deploy?

---

## 5. KESIMPULAN

Buatlah kesimpulan mencakup:
1. Pemahaman arsitektur CNN dan proses training.
2. Manfaat transfer learning dan data augmentation.
3. Perbandingan task: klasifikasi vs deteksi vs segmentasi.
4. Trade-off model (akurasi vs kecepatan vs ukuran).
5. Pentingnya format ONNX untuk deployment.
6. Perbandingan metode deteksi klasik (HOG) vs deep learning.
7. Peran loss function dan optimizer dalam keberhasilan training.
8. Efektivitas teknik regularisasi (Batch Normalization, Dropout).
9. Pentingnya metrik evaluasi yang komprehensif (precision, recall, F1, mAP, IoU).
10. Integrasi konsep end-to-end dari data preparation hingga deployment.
