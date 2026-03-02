# JOBSHEET PRAKTIKUM
# MODUL 9: DEEP LEARNING UNTUK KOMPUTER VISION

---

## 1. TUJUAN PRAKTIKUM

Setelah menyelesaikan praktikum ini, mahasiswa mampu:
1. Memahami preprocessing gambar untuk model DNN menggunakan `cv2.dnn.blobFromImage`.
2. Melakukan klasifikasi gambar menggunakan model pre-trained via OpenCV DNN.
3. Membandingkan performa beberapa model klasifikasi (MobileNet, GoogLeNet, ResNet, VGG, EfficientNet).
4. Memvisualisasikan arsitektur CNN layer-by-layer (konvolusi, ReLU, pooling).
5. Memahami dan membandingkan fungsi aktivasi (ReLU, Sigmoid, Tanh, LeakyReLU, Softmax).
6. Mengimplementasikan operasi konvolusi dan pooling secara manual.
7. Menerapkan data augmentasi dasar (flip, rotasi, brightness, noise, crop).
8. Memahami konsep transfer learning (scratch, feature extraction, fine-tuning).
9. Memvisualisasikan proses backpropagation dan gradient descent.
10. Mengimplementasikan deteksi objek dengan sliding window dan image pyramid.
11. Mengimplementasikan deteksi objek menggunakan HOG (Histogram of Oriented Gradients).
12. Memahami konsep grid, anchor boxes, dan IoU pada arsitektur YOLO.
13. Mengimplementasikan semantic segmentation manual (HSV, K-Means, overlay).
14. Memahami konsep instance segmentation dan perbedaannya dengan semantic segmentation.
15. Memvisualisasikan berbagai loss function (MSE, BCE, CCE, Dice, Focal).
16. Membandingkan perilaku optimizer (SGD, Momentum, Adam, RMSProp).
17. Memahami efek Batch Normalization dan Dropout pada training.
18. Menghitung dan memvisualisasikan metrik evaluasi model (Confusion Matrix, ROC, PR Curve).
19. Memahami konsep ONNX export, quantization, dan deployment pipeline.
20. Membuat proyek klasifikasi bentuk end-to-end sebagai integrasi seluruh konsep.

---

## 2. ALAT DAN BAHAN

### Perangkat Keras
- Laptop/PC (GPU NVIDIA direkomendasikan, CPU juga bisa)

### Perangkat Lunak
- Python 3.8+, VS Code
- Library: `opencv-python`, `numpy`, `matplotlib`

### Dataset
- Gambar dari folder `image/` (diunduh via `download_image.py`)
- Gambar sintetis yang di-generate oleh program

---

## 3. LANGKAH KERJA

### Percobaan 1: OpenCV DNN Blob — Preprocessing untuk Deep Learning

**Tujuan**: Memahami fungsi `cv2.dnn.blobFromImage` untuk preprocessing gambar sebelum inferensi.

**Langkah Kerja**:
1. Buat file `01_opencv_dnn_blob.py`.
2. Load gambar dari folder `image/`.
3. Buat blob dengan `cv2.dnn.blobFromImage(img, scalefactor, size, mean, swapRB)`.
4. Tampilkan parameter blob: shape, dtype, min, max.
5. Visualisasikan channel blob secara terpisah.
6. Bandingkan preprocessing untuk berbagai model: MobileNet, ResNet, VGG.
7. Tampilkan gambar asli vs blob channel.
8. Simpan hasil ke folder `output/`.

---

### Percobaan 2: Klasifikasi Gambar dengan OpenCV DNN

**Tujuan**: Menggunakan model pre-trained untuk klasifikasi gambar tanpa framework DL berat.

**Langkah Kerja**:
1. Buat file `02_opencv_dnn_klasifikasi.py`.
2. Load model pre-trained dengan `cv2.dnn.readNet()` (jika tersedia, simulasi jika tidak).
3. Buat blob, `net.setInput(blob)`, `net.forward()`.
4. Parse output: temukan top-5 prediksi.
5. Klasifikasi berbagai gambar.
6. Ukur dan tampilkan waktu inferensi.
7. Tampilkan gambar + label + confidence.
8. Simpan hasil ke folder `output/`.

---

### Percobaan 3: Perbandingan Model Pre-trained

**Tujuan**: Membandingkan akurasi dan kecepatan beberapa model klasifikasi.

**Langkah Kerja**:
1. Buat file `03_perbandingan_model_pretrained.py`.
2. Simulasikan performa 5 model: MobileNetV2, GoogLeNet, ResNet50, VGG16, EfficientNet.
3. Bandingkan: akurasi top-1, parameter count, inference time, ukuran model.
4. Buat grafik radar chart perbandingan.
5. Visualisasikan trade-off akurasi vs kecepatan.
6. Benchmark pengaruh input size.
7. Simpan visualisasi ke folder `output/`.

---

### Percobaan 4: Visualisasi Arsitektur CNN

**Tujuan**: Memvisualisasikan cara kerja CNN layer-by-layer.

**Langkah Kerja**:
1. Buat file `04_arsitektur_cnn_visualisasi.py`.
2. Implementasikan pipeline CNN sederhana: konvolusi → ReLU → pooling.
3. Tampilkan output setiap layer secara visual.
4. Visualisasikan berbagai filter (edge, blur, sharpen).
5. Buat diagram arsitektur CNN terkenal (LeNet, VGG, ResNet).
6. Simpan semua visualisasi ke folder `output/`.

---

### Percobaan 5: Fungsi Aktivasi

**Tujuan**: Memahami dan membandingkan fungsi aktivasi yang digunakan dalam deep learning.

**Langkah Kerja**:
1. Buat file `05_fungsi_aktivasi.py`.
2. Implementasikan: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax.
3. Plot semua fungsi dan turunannya.
4. Visualisasikan efek aktivasi pada gambar.
5. Bandingkan sifat: range output, gradient, kelebihan/kekurangan.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 6: Operasi Konvolusi dan Pooling

**Tujuan**: Mengimplementasikan konvolusi dan pooling secara manual.

**Langkah Kerja**:
1. Buat file `06_konvolusi_dan_pooling.py`.
2. Terapkan berbagai kernel: Sobel, Laplacian, Gaussian, Sharpen.
3. Demonstrasikan efek stride dan padding.
4. Implementasikan Max Pooling dan Average Pooling manual.
5. Bandingkan MaxPool vs AvgPool pada gambar nyata.
6. Tampilkan dan simpan hasil ke folder `output/`.

---

### Percobaan 7: Data Augmentasi Dasar

**Tujuan**: Mengimplementasikan teknik augmentasi data untuk deep learning.

**Langkah Kerja**:
1. Buat file `07_data_augmentasi_dasar.py`.
2. Implementasikan: horizontal flip, rotasi random, brightness/contrast.
3. Tambah: Gaussian noise, random crop.
4. Visualisasikan 1 gambar asli + 8 versi augmentasi.
5. Demonstrasikan augmentasi untuk satu batch data.
6. Simpan grid augmentasi ke folder `output/`.

---

### Percobaan 8: Transfer Learning (Konsep)

**Tujuan**: Memahami 3 strategi transfer learning melalui visualisasi.

**Langkah Kerja**:
1. Buat file `08_transfer_learning_konsep.py`.
2. Buat diagram 3 strategi: Training from Scratch, Feature Extraction, Fine-tuning.
3. Simulasikan training curves untuk ketiga pendekatan.
4. Visualisasikan fitur yang dipelajari setiap layer (low-level → high-level).
5. Simpan visualisasi ke folder `output/`.

---

### Percobaan 9: Backpropagation Visualisasi

**Tujuan**: Memvisualisasikan proses forward pass, backpropagation, dan gradient descent.

**Langkah Kerja**:
1. Buat file `09_backpropagation_visualisasi.py`.
2. Implementasikan MLP sederhana untuk XOR problem.
3. Jalankan forward dan backward pass manual.
4. Visualisasikan loss curve dan gradient per layer.
5. Tampilkan decision boundary yang terbentuk.
6. Simpan semua visualisasi ke folder `output/`.

---

### Percobaan 10: Deteksi Objek dengan Sliding Window

**Tujuan**: Memahami konsep sliding window dan image pyramid untuk deteksi objek.

**Langkah Kerja**:
1. Buat file `10_deteksi_objek_sliding_window.py`.
2. Implementasikan sliding window generator.
3. Implementasikan image pyramid (multi-scale).
4. Implementasikan Non-Maximum Suppression (NMS).
5. Visualisasikan sliding window bergerak, pyramid level, dan NMS.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 11: Deteksi Objek dengan HOG

**Tujuan**: Mengimplementasikan deteksi objek menggunakan fitur HOG.

**Langkah Kerja**:
1. Buat file `11_deteksi_objek_hog.py`.
2. Hitung gradien gambar (magnitude + orientasi).
3. Visualisasikan HOG descriptor pada gambar.
4. Gunakan `cv2.HOGDescriptor` untuk deteksi pejalan kaki.
5. Demonstrasikan deteksi pada berbagai gambar.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 12: YOLO Konsep Grid

**Tujuan**: Memahami konsep grid, anchor boxes, dan IoU pada arsitektur YOLO.

**Langkah Kerja**:
1. Buat file `12_yolo_konsep_grid.py`.
2. Gambar grid overlay pada gambar.
3. Implementasikan perhitungan IoU.
4. Visualisasikan IoU pada dua bounding box.
5. Tampilkan anchor boxes pada beberapa cell.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 13: Semantic Segmentation Manual

**Tujuan**: Mengimplementasikan segmentasi semantik secara manual.

**Langkah Kerja**:
1. Buat file `13_semantic_segmentation_manual.py`.
2. Lakukan segmentasi berbasis warna (HSV range).
3. Lakukan segmentasi K-Means.
4. Overlay mask berwarna pada gambar asli.
5. Hitung persentase area per kelas.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 14: Instance Segmentation (Konsep)

**Tujuan**: Memahami perbedaan instance segmentation vs semantic segmentation.

**Langkah Kerja**:
1. Buat file `14_instance_segmentation_konsep.py`.
2. Buat gambar sintetis dengan beberapa objek.
3. Deteksi dan beri mask per-instance (connected components).
4. Beri warna unik per instance.
5. Buat perbandingan 4 task: klasifikasi, deteksi, semantic, instance segmentation.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 15: Loss Function Visualisasi

**Tujuan**: Memvisualisasikan dan memahami berbagai loss function.

**Langkah Kerja**:
1. Buat file `15_loss_function_visualisasi.py`.
2. Plot: MSE, Binary Cross-Entropy, Categorical Cross-Entropy, Dice Loss, Focal Loss.
3. Demonstrasikan perhitungan loss pada data sampel.
4. Visualisasikan 3D loss landscape.
5. Bandingkan perilaku loss untuk class imbalance.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 16: Optimizer Visualisasi

**Tujuan**: Membandingkan trajectory dan konvergensi berbagai optimizer.

**Langkah Kerja**:
1. Buat file `16_optimizer_visualisasi.py`.
2. Implementasikan SGD, SGD+Momentum, Adam, RMSProp secara manual.
3. Visualisasikan trajectory pada contour plot fungsi Rosenbrock.
4. Bandingkan kecepatan konvergensi.
5. Demonstrasikan efek learning rate.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 17: Batch Normalization dan Dropout

**Tujuan**: Memahami efek Batch Normalization dan Dropout pada training.

**Langkah Kerja**:
1. Buat file `17_batch_normalization_dropout.py`.
2. Simulasikan distribusi aktivasi tanpa dan dengan Batch Normalization.
3. Implementasikan Dropout secara manual (inverted dropout).
4. Visualisasikan neuron yang di-drop pada berbagai rate.
5. Simulasikan efek Dropout terhadap overfitting.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 18: Model Evaluasi Metrik

**Tujuan**: Menghitung dan memvisualisasikan metrik evaluasi model.

**Langkah Kerja**:
1. Buat file `18_model_evaluasi_metrik.py`.
2. Buat confusion matrix dari data prediksi simulasi.
3. Hitung Accuracy, Precision, Recall, dan F1-Score secara manual.
4. Visualisasikan confusion matrix sebagai heatmap.
5. Plot kurva ROC dan Precision-Recall untuk 3 model berbeda.
6. Tampilkan bar chart metrik per kelas.
7. Simpan visualisasi ke folder `output/`.

---

### Percobaan 19: ONNX dan Deployment Konsep

**Tujuan**: Memahami pipeline deployment model deep learning menggunakan ONNX.

**Langkah Kerja**:
1. Buat file `19_onnx_dan_deployment_konsep.py`.
2. Buat diagram pipeline: Training → Export → Optimize → Deploy → Inference.
3. Demonstrasikan blob preprocessing dengan OpenCV DNN.
4. Simulasikan quantization FP32 → INT8 dan ukur kompresi.
5. Benchmark waktu preprocessing pada berbagai ukuran input.
6. Simpan visualisasi ke folder `output/`.

---

### Percobaan 20: Proyek Klasifikasi Bentuk

**Tujuan**: Proyek akhir mengintegrasikan seluruh konsep Modul 9.

**Langkah Kerja**:
1. Buat file `20_proyek_klasifikasi_bentuk.py`.
2. Generate dataset sintetis: lingkaran, segitiga, persegi, bintang (4 kelas × 100 sampel).
3. Terapkan augmentasi (posisi random, noise).
4. Ekstraksi fitur: Hu Moments + kontur statistik (10 fitur).
5. Normalisasi fitur (z-score).
6. Klasifikasi KNN (k=5).
7. Evaluasi: confusion matrix, precision, recall.
8. Visualisasikan sampel prediksi (benar = hijau, salah = merah).
9. Simpan semua output ke folder `output/`.

---

## 4. ANALISIS

### Percobaan 1 — OpenCV DNN Blob
- Apa fungsi parameter scalefactor, mean, dan swapRB pada `blobFromImage`?
- Mengapa preprocessing yang berbeda diperlukan untuk model yang berbeda?

### Percobaan 2 — Klasifikasi DNN
- Apa keuntungan menggunakan OpenCV DNN dibanding framework DL penuh?
- Bagaimana confidence score dapat digunakan untuk filtering prediksi?

### Percobaan 3 — Perbandingan Model
- Model mana yang optimal untuk mobile/embedded deployment?
- Apa trade-off utama antara akurasi, kecepatan, dan ukuran model?

### Percobaan 4 — Arsitektur CNN
- Mengapa CNN menggunakan filter shared (parameter sharing)?
- Bagaimana feature map berubah dari layer awal ke layer dalam?

### Percobaan 5 — Fungsi Aktivasi
- Mengapa ReLU lebih populer dibanding Sigmoid untuk hidden layer?
- Apa masalah "dying ReLU" dan bagaimana LeakyReLU mengatasinya?

### Percobaan 6 — Konvolusi dan Pooling
- Apa perbedaan efek Max Pooling vs Average Pooling?
- Bagaimana stride dan padding mempengaruhi ukuran output?

### Percobaan 7 — Data Augmentasi
- Augmentasi mana yang paling berpengaruh terhadap performa model?
- Apakah augmentasi yang terlalu agresif bisa menurunkan performa?

### Percobaan 8 — Transfer Learning
- Kapan menggunakan feature extraction vs fine-tuning?
- Berapa minimum data yang diperlukan untuk fine-tuning yang efektif?

### Percobaan 9 — Backpropagation
- Bagaimana gradient mengalir mundur melalui jaringan?
- Apa yang terjadi jika learning rate terlalu besar atau terlalu kecil?

### Percobaan 10 — Sliding Window
- Mengapa sliding window tidak efisien untuk deteksi objek modern?
- Bagaimana NMS mengatasi masalah multiple detection?

### Percobaan 11 — HOG
- Bagaimana performa HOG dibanding deep learning-based detector?
- Pada kondisi apa HOG masih relevan digunakan?

### Percobaan 12 — YOLO Grid
- Mengapa YOLO membagi gambar menjadi grid?
- Bagaimana IoU threshold mempengaruhi kualitas deteksi?

### Percobaan 13 — Semantic Segmentation Manual
- Apa keterbatasan segmentasi manual dibanding model deep learning?
- Pada skenario apa segmentasi berbasis warna masih efektif?

### Percobaan 14 — Instance Segmentation
- Apa tantangan memisahkan instance yang saling berdekatan?
- Kapan instance segmentation lebih diperlukan dibanding semantic?

### Percobaan 15 — Loss Function
- Mengapa pemilihan loss function yang tepat sangat penting?
- Bagaimana Focal Loss mengatasi masalah class imbalance?

### Percobaan 16 — Optimizer
- Mengapa Adam lebih cepat konvergen dibanding SGD?
- Kapan SGD with momentum lebih baik dari Adam?

### Percobaan 17 — Batch Normalization dan Dropout
- Bagaimana BN mempercepat training?
- Apakah Dropout selalu meningkatkan performa?

### Percobaan 18 — Evaluasi Metrik
- Mengapa accuracy saja tidak cukup?
- Metrik mana yang paling penting untuk deteksi vs klasifikasi?

### Percobaan 19 — ONNX dan Deployment
- Apa keuntungan utama menggunakan ONNX?
- Bagaimana quantization mempengaruhi akurasi dan kecepatan?

### Percobaan 20 — Proyek Klasifikasi Bentuk
- Mengapa Hu Moments dipilih sebagai fitur?
- Bagaimana normalisasi fitur mempengaruhi hasil klasifikasi KNN?

---

## 5. KESIMPULAN

Buatlah kesimpulan mencakup:
1. Pemahaman preprocessing gambar untuk DNN (blob, normalisasi).
2. Perbandingan model pre-trained (MobileNet vs VGG vs ResNet).
3. Arsitektur CNN: konvolusi, pooling, aktivasi, batch norm.
4. Teknik transfer learning dan data augmentasi.
5. Metode deteksi: sliding window, HOG, YOLO.
6. Segmentasi: semantic vs instance.
7. Loss functions dan optimizer dalam training.
8. Teknik regularisasi: Batch Normalization dan Dropout.
9. Metrik evaluasi: Confusion Matrix, Precision, Recall, F1, ROC-AUC.
10. Pipeline deployment: ONNX, quantization, OpenCV DNN.
