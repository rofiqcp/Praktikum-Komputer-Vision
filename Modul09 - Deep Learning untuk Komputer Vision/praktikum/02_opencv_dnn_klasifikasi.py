"""
==========================================================================
PERCOBAAN 2: KLASIFIKASI GAMBAR DENGAN OPENCV DNN
==========================================================================
Program ini mempelajari cara melakukan klasifikasi gambar menggunakan
model pre-trained (MobileNet/GoogLeNet) melalui modul cv2.dnn.
Model di-load dalam format Caffe/ONNX, lalu digunakan untuk inferensi.

Fungsi utama:
- cv2.dnn.readNet()        : Memuat model neural network
- net.setInput(blob)       : Memasukkan blob ke network
- net.forward()            : Menjalankan inferensi (forward pass)
- Interpretasi output probabilitas per kelas

Referensi: Mastering OpenCV 4, Ch.13
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gambar(nama_file):
    """Memuat gambar dari folder image/."""
    path = os.path.join(IMAGE_DIR, nama_file)
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] {nama_file} tidak ditemukan!")
    return img


def load_labels_imagenet():
    """
    Memuat label kelas ImageNet (1000 kelas).
    Jika file tidak ada, gunakan label dummy untuk demonstrasi.
    """
    label_path = os.path.join(MODEL_DIR, "synset_words.txt")
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        return labels
    # Label dummy jika file tidak tersedia
    return [f"Kelas_{i}" for i in range(1000)]


def klasifikasi_dengan_dnn(img, labels):
    """
    Melakukan klasifikasi gambar menggunakan OpenCV DNN.
    Menggunakan model GoogLeNet (Caffe) jika tersedia.
    Jika model tidak ada, simulasi dengan fitur warna sederhana.
    """
    model_path = os.path.join(MODEL_DIR, "bvlc_googlenet.caffemodel")
    config_path = os.path.join(MODEL_DIR, "bvlc_googlenet.prototxt")

    if os.path.exists(model_path) and os.path.exists(config_path):
        # Load model GoogLeNet dari Caffe
        net = cv2.dnn.readNet(model_path, config_path)
        # Buat blob: resize 224x224, normalisasi, mean subtraction
        blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123), False, False)
        net.setInput(blob)
        start = time.time()
        output = net.forward()
        waktu = (time.time() - start) * 1000
        # Ambil top-5 prediksi
        top5_idx = np.argsort(output[0])[::-1][:5]
        hasil = [(labels[i], output[0][i]) for i in top5_idx]
        return hasil, waktu
    else:
        print("  [INFO] Model tidak tersedia, menggunakan simulasi klasifikasi")
        # Simulasi berdasarkan statistik warna
        mean_color = np.mean(img, axis=(0, 1))
        np.random.seed(int(mean_color.sum()) % 1000)
        probs = np.random.dirichlet(np.ones(5))
        sample_labels = ["kucing", "anjing", "mobil", "bunga", "gedung"]
        hasil = list(zip(sample_labels, sorted(probs, reverse=True)))
        return hasil, 0.0


def visualisasi_hasil_klasifikasi(img, hasil, nama_output):
    """
    Menampilkan gambar dengan hasil klasifikasi top-5.
    Menyimpan visualisasi sebagai bar chart horizontal.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Tampilkan gambar
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Gambar Input")
    ax1.axis('off')

    # Bar chart probabilitas top-5
    labels = [h[0][:30] for h in hasil]
    probs = [h[1] * 100 for h in hasil]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    ax2.barh(range(len(labels)), probs, color=colors)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Probabilitas (%)")
    ax2.set_title("Top-5 Prediksi")
    ax2.invert_yaxis()

    plt.suptitle("Klasifikasi Gambar dengan OpenCV DNN", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, nama_output)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] output/{nama_output}")


def klasifikasi_batch(daftar_gambar, labels):
    """
    Mengklasifikasi beberapa gambar sekaligus dan membandingkan hasilnya.
    Menyimpan hasil perbandingan dalam satu gambar grid.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flat

    for i, nama in enumerate(daftar_gambar[:6]):
        img = load_gambar(nama)
        if img is None:
            continue
        hasil, waktu = klasifikasi_dengan_dnn(img, labels)
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        top_label = hasil[0][0] if hasil else "?"
        top_prob = hasil[0][1] * 100 if hasil else 0
        axes[i].set_title(f"{top_label}\n{top_prob:.1f}% ({waktu:.0f}ms)", fontsize=9)
        axes[i].axis('off')

    plt.suptitle("Klasifikasi Batch - OpenCV DNN", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_batch_klasifikasi.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [SAVED] output/02_batch_klasifikasi.png")


def main():
    """Fungsi utama: klasifikasi gambar dengan OpenCV DNN module."""
    print("=" * 60)
    print("PERCOBAAN 2: KLASIFIKASI GAMBAR DENGAN OPENCV DNN")
    print("=" * 60)

    labels = load_labels_imagenet()

    # 1. Klasifikasi satu gambar
    print("\n--- 1. Klasifikasi Satu Gambar ---")
    img = load_gambar("kucing.jpg")
    if img is not None:
        hasil, waktu = klasifikasi_dengan_dnn(img, labels)
        for label, prob in hasil:
            print(f"  {label}: {prob * 100:.2f}%")
        if waktu > 0:
            print(f"  Waktu inferensi: {waktu:.1f} ms")
        visualisasi_hasil_klasifikasi(img, hasil, "02_klasifikasi_hasil.png")

        # Tampilkan gambar
        cv2.imshow("Klasifikasi DNN - Percobaan 2", cv2.resize(img, (400, 400)))
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    # 2. Klasifikasi batch
    print("\n--- 2. Klasifikasi Batch ---")
    daftar = ["kucing.jpg", "anjing.jpg", "mobil.jpg", "bunga.jpg", "gedung.jpg"]
    klasifikasi_batch(daftar, labels)

    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 2")
    print("=" * 60)
    print("""
Konsep yang dipelajari:
1. cv2.dnn.readNet() memuat model dalam format Caffe/ONNX/TF
2. net.setInput(blob) + net.forward() melakukan inferensi
3. Output berupa probabilitas per kelas (softmax)
4. Top-5 prediksi menunjukkan 5 kelas paling mungkin
5. Waktu inferensi bergantung pada model dan ukuran input

Output: output/02_klasifikasi_hasil.png, output/02_batch_klasifikasi.png
""")


if __name__ == "__main__":
    main()
