"""
==========================================================================
PERCOBAAN 19: ONNX DAN DEPLOYMENT KONSEP
==========================================================================
Program ini mendemonstrasikan konsep deployment model deep learning
menggunakan format ONNX (Open Neural Network Exchange).

Konsep yang dipelajari:
- Format ONNX: standar terbuka untuk representasi model
- Konversi model ke ONNX
- Inference menggunakan OpenCV DNN dengan model ONNX
- Optimasi model (quantization konsep)
- Pipeline deployment: training -> export -> optimize -> deploy

Referensi: Szeliski Ch.5, ONNX Runtime docs
==========================================================================
"""

import cv2
import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gambar(nama_file):
    """Memuat gambar dari folder image/."""
    path = os.path.join(IMAGE_DIR, nama_file)
    img = cv2.imread(path)
    if img is None:
        print(f"  [WARN] Gambar {nama_file} tidak ditemukan.")
        return None
    return img


def simulasi_onnx_pipeline():
    """
    Menampilkan pipeline deployment model:
    Training -> Export ONNX -> Optimize -> Deploy (Inference).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    stages = [
        (1.0, 2.0, "1. TRAINING\n(PyTorch/TF)", "#4ECDC4"),
        (3.0, 2.0, "2. EXPORT\nke ONNX", "#FF6B6B"),
        (5.0, 2.0, "3. OPTIMIZE\n(Quantize)", "#45B7D1"),
        (7.0, 2.0, "4. DEPLOY\n(OpenCV DNN)", "#96CEB4"),
        (9.0, 2.0, "5. INFERENCE\n(Produksi)", "#FFEAA7"),
    ]

    for x, y, label, color in stages:
        rect = plt.Rectangle((x - 0.8, y - 0.6), 1.6, 1.2, facecolor=color,
                              edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

    # Panah
    for i in range(len(stages) - 1):
        ax.annotate('', xy=(stages[i + 1][0] - 0.8, 2.0),
                     xytext=(stages[i][0] + 0.8, 2.0),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Keuntungan ONNX
    benefits = [
        "Interoperabilitas", "Portabilitas", "Optimasi Hardware",
        "Framework-Agnostic", "Edge Computing"
    ]
    for i, b in enumerate(benefits):
        ax.text(2 + i * 1.5, 0.5, f"✓ {b}", fontsize=8, ha='center',
                bbox=dict(facecolor='lightyellow', edgecolor='gray', boxstyle='round'))

    ax.set_title("Pipeline Deployment Model Deep Learning dengan ONNX", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "19_onnx_pipeline.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/19_onnx_pipeline.png")


def demo_inference_opencv_dnn(img):
    """
    Demonstrasi inference dengan OpenCV DNN module.
    Menggunakan blob preprocessing dan forward pass.
    """
    if img is None:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print("  [INFO] Menggunakan gambar random karena gambar tidak tersedia")

    print("  Preprocessing gambar untuk inference:")
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (224, 224),
                                  (0.485, 0.456, 0.406), swapRB=True, crop=False)
    print(f"    Input shape: {img.shape}")
    print(f"    Blob shape:  {blob.shape}")
    print(f"    Blob dtype:  {blob.dtype}")
    print(f"    Blob range:  [{blob.min():.4f}, {blob.max():.4f}]")

    return blob


def simulasi_quantization():
    """
    Simulasi konsep quantization:
    Mengubah weight dari float32 ke int8 untuk memperkecil ukuran model
    dan mempercepat inference (dengan sedikit kehilangan akurasi).
    """
    np.random.seed(42)

    # Simulasi weight
    weights_fp32 = np.random.randn(1000).astype(np.float32)

    # Quantize ke int8
    scale = (weights_fp32.max() - weights_fp32.min()) / 255
    zero_point = int(-weights_fp32.min() / scale)
    weights_int8 = np.clip(np.round(weights_fp32 / scale) + zero_point, 0, 255).astype(np.uint8)

    # Dequantize
    weights_dequant = (weights_int8.astype(np.float32) - zero_point) * scale

    # Error
    error = np.abs(weights_fp32 - weights_dequant)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Distribusi float32
    axes[0, 0].hist(weights_fp32, bins=50, color='steelblue', alpha=0.7)
    axes[0, 0].set_title(f"Float32 Weights\nSize: {weights_fp32.nbytes} bytes")

    # Distribusi int8
    axes[0, 1].hist(weights_int8, bins=50, color='coral', alpha=0.7)
    axes[0, 1].set_title(f"Int8 Quantized\nSize: {weights_int8.nbytes} bytes")

    # Perbandingan
    idx = np.argsort(weights_fp32)[:100]
    axes[1, 0].plot(weights_fp32[idx], 'b-', label='FP32 Original', linewidth=1)
    axes[1, 0].plot(weights_dequant[idx], 'r--', label='Dequantized', linewidth=1)
    axes[1, 0].set_title("Perbandingan Weight")
    axes[1, 0].legend()

    # Error distribusi
    axes[1, 1].hist(error, bins=50, color='green', alpha=0.7)
    axes[1, 1].set_title(f"Quantization Error\nMean: {error.mean():.6f}")

    compression = weights_fp32.nbytes / weights_int8.nbytes
    plt.suptitle(f"Quantization: FP32 → INT8 (Kompresi {compression:.1f}x)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "19_quantization.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/19_quantization.png")


def benchmark_inference():
    """
    Benchmark waktu inference untuk berbagai ukuran input.
    Simulasi overhead preprocessing dan forward pass.
    """
    sizes = [(128, 128), (224, 224), (416, 416), (640, 640)]
    times_preprocessing = []
    times_forward = []

    for h, w in sizes:
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Preprocessing time
        start = time.time()
        for _ in range(10):
            blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (224, 224), swapRB=True)
        t_prep = (time.time() - start) / 10 * 1000
        times_preprocessing.append(t_prep)

        # Simulate forward time (proportional to input size)
        t_fwd = (h * w) / (224 * 224) * 15 + np.random.uniform(1, 3)
        times_forward.append(t_fwd)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width / 2, times_preprocessing, width, label='Preprocessing', color='steelblue')
    ax.bar(x + width / 2, times_forward, width, label='Forward Pass (sim)', color='coral')

    ax.set_xlabel("Input Size")
    ax.set_ylabel("Waktu (ms)")
    ax.set_title("Benchmark Inference Time vs Input Size")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}x{w}" for h, w in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "19_benchmark.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  [SAVED] output/19_benchmark.png")


def main():
    """Fungsi utama: ONNX dan deployment konsep."""
    print("=" * 60)
    print("PERCOBAAN 19: ONNX DAN DEPLOYMENT KONSEP")
    print("=" * 60)

    print("\n--- 1. Pipeline Deployment ONNX ---")
    simulasi_onnx_pipeline()

    print("\n--- 2. Demo Inference OpenCV DNN ---")
    img = load_gambar("kucing.jpg")
    demo_inference_opencv_dnn(img)

    print("\n--- 3. Konsep Quantization ---")
    simulasi_quantization()

    print("\n--- 4. Benchmark Inference ---")
    benchmark_inference()

    print("\n" + "=" * 60)
    print("RINGKASAN PERCOBAAN 19")
    print("=" * 60)
    print("""
Konsep deployment model deep learning:
1. ONNX: format standar terbuka untuk representasi model
2. Pipeline: Train -> Export -> Optimize -> Deploy
3. OpenCV DNN: modul bawaan untuk inference tanpa framework berat
4. Quantization: kompresi model FP32->INT8 (4x lebih kecil)
5. Preprocessing blob: standar input untuk DNN module
6. Trade-off: ukuran model vs akurasi vs kecepatan

Output: output/19_onnx_pipeline.png, output/19_quantization.png,
        output/19_benchmark.png
""")


if __name__ == "__main__":
    main()
