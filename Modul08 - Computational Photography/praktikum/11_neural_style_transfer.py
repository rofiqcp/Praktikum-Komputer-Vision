"""
==========================================================================
PERCOBAAN 11: NEURAL STYLE TRANSFER (NST)
==========================================================================
Program ini mempelajari neural style transfer (nst).
Praktikum 11 - Neural Style Transfer (NST)
Modul 08: Computational Photography

Topik: Gram matrix, content + style loss, cv2.stylization(), artistik filter
Referensi: Mastering OpenCV 4 with Python Ch.12,
           Hands-On ML with Scikit-Learn, Keras & TF 2nd Ed Ch.17 (Géron),
           Deep Learning for CV with Python (Rosebrock)

Hasil: Visualisasi dan analisis disimpan ke folder output/
==========================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)



def explain_nst_theory():
    """Penjelasan konsep Neural Style Transfer."""
    print("=" * 58)
    print("KONSEP NEURAL STYLE TRANSFER (NST)")
    print("=" * 58)
    print("""
Gatys et al. (2015) menemukan bahwa:
  - CNN layers AWAL  → menangkap KONTEN (struktur, bentuk)
  - CNN layers DALAM → menangkap GAYA (tekstur, pola)

Komponen NST:
  ┌─────────────────────────────────────────────────────┐
  │  Content Image  ─┐                                  │
  │                  ├──► CNN (VGG-19) ──► Loss Total   │
  │  Style Image    ─┘                                  │
  │                                                     │
  │  Content Loss: ||F_content - F_generated||²         │
  │  Style Loss:   ||G_style - G_generated||²           │
  │    → G = Gram Matrix = F·Fᵀ (tangkap korelasi)     │
  │                                                     │
  │  Total Loss = α·Content_Loss + β·Style_Loss         │
  └─────────────────────────────────────────────────────┘

Implementasi modern via OpenCV DNN (model terlatih):
  - Faster Artistic Style Transfer (Johnson 2016)
  - Satu model per style, inference real-time
""")


def demo_opencv_artistic_filters():
    """Filter artistik bawaan OpenCV (tanpa model DNN eksternal)."""
    # Buat gambar konten sintetis yang menarik
    content = np.zeros((300, 400, 3), dtype=np.uint8)
    # Langit
    for y in range(150):
        b = int(200 + y * 0.4)
        g = int(150 + y * 0.3)
        r = int(100 + y * 0.2)
        content[y, :] = [b, g, r]
    # Tanah
    content[150:, :] = [50, 120, 60]
    # Pohon
    cv2.circle(content, (150, 120), 50, (30, 100, 30), -1)
    cv2.rectangle(content, (140, 160), (160, 200), (60, 40, 20), -1)
    cv2.circle(content, (300, 110), 40, (20, 80, 20), -1)
    cv2.rectangle(content, (293, 145), (308, 185), (50, 35, 15), -1)
    # Matahari
    cv2.circle(content, (50, 50), 30, (0, 200, 255), -1)

    print("\nMenerapkan artistic filters bawaan OpenCV:")

    # 1. Stylization (oil paint / cartoon look)
    stylized = cv2.stylization(content, sigma_s=60, sigma_r=0.45)
    print("  cv2.stylization() — sigma_s=60, sigma_r=0.45")

    # 2. Pencil Sketch (grayscale)
    gray_sketch, _ = cv2.pencilSketch(content, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    print("  cv2.pencilSketch() grayscale — sigma_s=60, sigma_r=0.07")

    # 3. Pencil Sketch (color)
    _, color_sketch = cv2.pencilSketch(content, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    print("  cv2.pencilSketch() color — shade_factor=0.05")

    # 4. Detail Enhance
    detail = cv2.detailEnhance(content, sigma_s=10, sigma_r=0.15)
    print("  cv2.detailEnhance() — sigma_s=10, sigma_r=0.15")

    # 5. Edge Preserving Filter
    edge_pres = cv2.edgePreservingFilter(content, flags=1, sigma_s=60, sigma_r=0.4)
    print("  cv2.edgePreservingFilter() — flags=RECURS_FILTER")

    plt.figure(figsize=(18, 6))
    panels = [
        ("Konten Asli", content),
        ("Stylization\n(Oil Paint Look)", stylized),
        ("Pencil Sketch\n(Grayscale)", cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2RGB)),
        ("Pencil Sketch\n(Color)", color_sketch),
        ("Detail Enhance", detail),
        ("Edge Preserving\nFilter", edge_pres),
    ]
    for i, (title, img) in enumerate(panels):
        plt.subplot(1, len(panels), i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        elif img.shape[2] == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype == np.uint8 else img)
        else:
            plt.imshow(img)
        plt.title(title, fontsize=9); plt.axis('off')
    plt.suptitle("Artistic Filters Bawaan OpenCV (Non-Photorealistic Rendering)")
    plt.tight_layout(); plt.savefig("output_11_artistic_filters.png", dpi=100); plt.show()


def demo_nst_gram_matrix():
    """Demonstrasi Gram Matrix sebagai style representation."""
    print("\n[Gram Matrix] Cara kerja style representation:")
    print("  - Gram matrix G = F·Fᵀ, di mana F = feature maps")
    print("  - G[i,j] = dot product antara filter i dan filter j")
    print("  - Menangkap korelasi antar fitur (tekstur, pola)")
    print("  - Invariant terhadap posisi spasial")

    # Simulasi feature maps 4 channel
    np.random.seed(42)
    F = np.random.randn(4, 16).astype(np.float32)  # 4 filters, 16 spatial locations
    G = F @ F.T  # Gram matrix 4×4

    print(f"\n  Feature map shape: {F.shape} (4 filters × 16 spatial)")
    print(f"  Gram matrix shape: {G.shape}")
    print(f"  Gram matrix:\n{G.round(2)}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(F, cmap='RdBu', aspect='auto')
    plt.colorbar(); plt.title("Feature Maps F\n(4 filters × 16 spatial)"); plt.xlabel("Spatial locations"); plt.ylabel("Filter index")
    plt.subplot(1, 2, 2)
    plt.imshow(G, cmap='viridis')
    plt.colorbar(); plt.title("Gram Matrix G = F·Fᵀ\n(Style Representation 4×4)")
    plt.xlabel("Filter j"); plt.ylabel("Filter i")
    plt.tight_layout(); plt.savefig("output_11_gram_matrix.png", dpi=100); plt.show()


def demo_nst_parameter_effect():
    """Pengaruh parameter sigma pada stylization."""
    content = np.zeros((250, 350, 3), dtype=np.uint8)
    for y in range(125):
        content[y, :] = [int(200 + y * 0.4), int(180 + y * 0.2), 100]
    content[125:, :] = [50, 120, 60]
    cv2.circle(content, (120, 100), 50, (30, 90, 30), -1)
    cv2.rectangle(content, (110, 140), (130, 175), (60, 40, 20), -1)

    configs = [
        ("σ_s=10, σ_r=0.1\n(Sedikit stilisasi)", 10, 0.1),
        ("σ_s=60, σ_r=0.4\n(Sedang)", 60, 0.4),
        ("σ_s=150, σ_r=0.8\n(Sangat stilisasi)", 150, 0.8),
    ]

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1); plt.imshow(cv2.cvtColor(content, cv2.COLOR_BGR2RGB)); plt.title("Asli"); plt.axis('off')
    for i, (title, ss, sr) in enumerate(configs):
        result = cv2.stylization(content, sigma_s=ss, sigma_r=sr)
        plt.subplot(1, 4, i + 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=9); plt.axis('off')
    plt.suptitle("Pengaruh Parameter sigma_s dan sigma_r pada Stylization")
    plt.tight_layout(); plt.savefig("output_11_nst_parameters.png", dpi=100); plt.show()


def demo_nst_dnn_pipeline():
    """Penjelasan pipeline NST dengan pretrained DNN model."""
    print("\n[Pipeline NST dengan Pretrained Model]")
    print("""
Pipeline: Fast Style Transfer (Johnson 2016) via OpenCV DNN

1. Download model terlatih (.t7 atau ONNX):
   - udnie.t7, mosaic.t7, candy.t7, feathers.t7
   - Sumber: https://cs.stanford.edu/people/jcjohns/fast-neural-style/

2. Load dengan OpenCV DNN:
   net = cv2.dnn.readNetFromTorch("mosaic.t7")

3. Preprocessing:
   blob = cv2.dnn.blobFromImage(
       img, 1.0, (width, height),
       (103.939, 116.779, 123.680), swapRB=False
   )

4. Inference:
   net.setInput(blob)
   output = net.forward()

5. Postprocessing:
   output = output.reshape(3, height, width)
   output += np.array([103.939, 116.779, 123.680]).reshape(3, 1, 1)
   output = np.clip(output, 0, 255).astype(np.uint8)
   result = output.transpose(1, 2, 0)[:, :, ::-1]

Catatan: Model DNN tidak disertakan, gunakan cv2.stylization()
sebagai alternatif built-in yang tidak butuh model eksternal.
""")

    # Perbandingan waktu processing
    content = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    import time
    times = {}
    for name, fn in [
        ("stylization", lambda: cv2.stylization(content, sigma_s=60, sigma_r=0.45)),
        ("pencilSketch", lambda: cv2.pencilSketch(content, sigma_s=60, sigma_r=0.07, shade_factor=0.05)),
        ("detailEnhance", lambda: cv2.detailEnhance(content, sigma_s=10, sigma_r=0.15)),
        ("edgePreservingFilter", lambda: cv2.edgePreservingFilter(content, flags=1, sigma_s=60, sigma_r=0.4)),
    ]:
        t0 = time.time()
        for _ in range(10):
            fn()
        times[name] = (time.time() - t0) / 10 * 1000
        print(f"  {name}: {times[name]:.1f}ms/frame ({1000/times[name]:.1f} FPS)")

    plt.bar(times.keys(), times.values(), color=['steelblue', 'coral', 'lightgreen', 'gold'])
    plt.ylabel("Waktu (ms)"); plt.title("Kecepatan Built-in Artistic Filters OpenCV")
    plt.xticks(rotation=20); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig("output_11_nst_timing.png", dpi=100); plt.show()


if __name__ == "__main__":
    print("=" * 55)
    print("PRAKTIKUM 11: NEURAL STYLE TRANSFER")
    print("=" * 55)

    explain_nst_theory()

    print("\n[1] Artistic Filters Bawaan OpenCV")
    demo_opencv_artistic_filters()

    print("\n[2] Gram Matrix — Style Representation")
    demo_nst_gram_matrix()

    print("\n[3] Pengaruh Parameter Sigma pada Stylization")
    demo_nst_parameter_effect()

    print("\n[4] Pipeline NST dengan Pretrained DNN")
    demo_nst_dnn_pipeline()

    print("\n[SELESAI] Semua demo Neural Style Transfer berhasil dijalankan.")
