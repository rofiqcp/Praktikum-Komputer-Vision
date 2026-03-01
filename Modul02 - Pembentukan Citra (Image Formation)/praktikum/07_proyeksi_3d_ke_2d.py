"""
==========================================================================
PERCOBAAN 07: PROYEKSI 3D KE 2D
==========================================================================
Mempelajari cara memproyeksikan titik 3D ke bidang gambar 2D
menggunakan parameter kamera (intrinsik dan ekstrinsik).

Fungsi:
- cv2.projectPoints() → Proyeksi 3D→2D
- cv2.Rodrigues() → Konversi vektor rotasi ↔ matriks rotasi
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 07: PROYEKSI 3D KE 2D")
print("=" * 60)

# ============================================================
# 1. Definisi parameter kamera (simulasi)
# ============================================================
print("\n--- 1. Parameter Kamera ---")

# Matriks kamera intrinsik:
# [[fx, 0, cx],
#  [0, fy, cy],
#  [0,  0,  1]]
fx = fy = 500.0  # Focal length dalam piksel
cx, cy = 320, 240  # Principal point (pusat gambar)

camera_matrix = np.float64([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])
print(f"  Focal length: {fx}")
print(f"  Principal point: ({cx}, {cy})")

# Koefisien distorsi (tanpa distorsi)
dist_coeffs = np.zeros(5)

# ============================================================
# 2. Definisi objek 3D (kubus)
# ============================================================
print("\n--- 2. Objek 3D: Kubus ---")

# 8 titik sudut kubus (ukuran 2×2×2, pusat di origin)
kubus_3d = np.float64([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Sisi depan
    [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],   # Sisi belakang
])
print(f"  Kubus: 8 titik, ukuran 2×2×2")

# Garis penghubung kubus (pasangan indeks)
edges = [
    (0,1),(1,2),(2,3),(3,0),  # Depan
    (4,5),(5,6),(6,7),(7,4),  # Belakang
    (0,4),(1,5),(2,6),(3,7),  # Penghubung
]

# ============================================================
# 3. Proyeksi dari berbagai sudut pandang
# ============================================================
print("\n--- 3. Berbagai Sudut Pandang ---")

# Vektor rotasi (Rodrigues) dan vektor translasi
sudut_pandang = [
    ("Depan",         [0, 0, 0],       [0, 0, 5]),
    ("Atas",          [-0.5, 0, 0],    [0, 0, 6]),
    ("Samping-Kanan", [0, 0.5, 0],     [0, 0, 6]),
    ("Miring",        [0.3, 0.3, 0.1], [0, 0, 5]),
    ("Perspektif",    [0.4, -0.3, 0],  [0.5, 0.5, 6]),
    ("Dekat",         [0.2, 0.2, 0],   [0, 0, 3]),
]

hasil_pandang = []

for nama, rvec, tvec in sudut_pandang:
    rvec = np.float64(rvec)
    tvec = np.float64(tvec)

    # cv2.projectPoints: proyeksi 3D→2D
    # Input: titik 3D, vektor rotasi, vektor translasi, camera matrix, distorsi
    # Output: titik 2D, jacobian
    pts_2d, _ = cv2.projectPoints(kubus_3d, rvec, tvec, camera_matrix, dist_coeffs)

    # hasil pts_2d berbentuk (N, 1, 2), reshape ke (N, 2)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    hasil_pandang.append((nama, pts_2d))
    print(f"  {nama}: titik 2D range x=[{pts_2d[:,0].min()},{pts_2d[:,0].max()}]")

# ============================================================
# 4. Menggambar wireframe kubus
# ============================================================
print("\n--- 4. Wireframe Rendering ---")

def gambar_kubus(pts_2d, nama, img_size=(640, 480)):
    """Menggambar wireframe kubus dari titik 2D yang diproyeksikan."""
    canvas = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 240

    # Gambar garis penghubung (edges)
    warna_edge = [
        (200, 0, 0), (200, 0, 0), (200, 0, 0), (200, 0, 0),     # Depan: biru
        (0, 0, 200), (0, 0, 200), (0, 0, 200), (0, 0, 200),     # Belakang: merah
        (0, 150, 0), (0, 150, 0), (0, 150, 0), (0, 150, 0),     # Penghubung: hijau
    ]
    for i, (a, b) in enumerate(edges):
        pt1 = tuple(pts_2d[a])
        pt2 = tuple(pts_2d[b])
        cv2.line(canvas, pt1, pt2, warna_edge[i], 2)

    # Gambar titik sudut
    for i, pt in enumerate(pts_2d):
        cv2.circle(canvas, tuple(pt), 5, (0, 0, 255), -1)
        cv2.putText(canvas, str(i), tuple(pt + [5, -5]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.putText(canvas, nama, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return canvas

# ============================================================
# 5. cv2.Rodrigues: konversi vektor rotasi ↔ matriks rotasi
# ============================================================
print("\n--- 5. Rodrigues ---")

rvec_test = np.float64([0.3, 0.3, 0.1])
# cv2.Rodrigues mengkonversi vektor rotasi (3×1) → matriks rotasi (3×3)
R, _ = cv2.Rodrigues(rvec_test)
print(f"  Vektor rotasi: {rvec_test}")
print(f"  Matriks rotasi 3×3:\n{R}")

# Konversi balik: matriks → vektor
rvec_back, _ = cv2.Rodrigues(R)
print(f"  Vektor kembali: {rvec_back.flatten()}")

# ============================================================
# 6. Proyeksi sumbu koordinat 3D
# ============================================================
print("\n--- 6. Sumbu Koordinat ---")

# Titik origin dan ujung sumbu X, Y, Z
axis_pts = np.float64([
    [0, 0, 0],  # Origin
    [2, 0, 0],  # Ujung X
    [0, 2, 0],  # Ujung Y
    [0, 0, 2],  # Ujung Z
])

rvec_axis = np.float64([0.3, 0.3, 0])
tvec_axis = np.float64([0, 0, 5])
axis_2d, _ = cv2.projectPoints(axis_pts, rvec_axis, tvec_axis, camera_matrix, dist_coeffs)
axis_2d = axis_2d.reshape(-1, 2).astype(int)

# Buat gambar sumbu
canvas_axis = np.ones((480, 640, 3), dtype=np.uint8) * 240
origin = tuple(axis_2d[0])
cv2.arrowedLine(canvas_axis, origin, tuple(axis_2d[1]), (0, 0, 255), 3)  # X = merah
cv2.arrowedLine(canvas_axis, origin, tuple(axis_2d[2]), (0, 255, 0), 3)  # Y = hijau
cv2.arrowedLine(canvas_axis, origin, tuple(axis_2d[3]), (255, 0, 0), 3)  # Z = biru
cv2.putText(canvas_axis, "X", tuple(axis_2d[1] + [10, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(canvas_axis, "Y", tuple(axis_2d[2] + [10, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(canvas_axis, "Z", tuple(axis_2d[3] + [10, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# ============================================================
# 7. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, (nama, pts) in enumerate(hasil_pandang[:6]):
    r, c = divmod(i, 4)
    if i < 6:
        canvas = gambar_kubus(pts, nama)
        ax = axes[r, c] if i < 4 else axes[1, i - 4]
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(nama)
        ax.axis("off")

axes[1, 2].imshow(cv2.cvtColor(canvas_axis, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("Sumbu 3D")
axes[1, 2].axis("off")

# Info teks
axes[1, 3].text(0.1, 0.7, f"fx={fx}, fy={fy}", fontsize=12, transform=axes[1,3].transAxes)
axes[1, 3].text(0.1, 0.5, f"cx={cx}, cy={cy}", fontsize=12, transform=axes[1,3].transAxes)
axes[1, 3].text(0.1, 0.3, "Merah=X, Hijau=Y", fontsize=12, transform=axes[1,3].transAxes)
axes[1, 3].text(0.1, 0.1, "Biru=Z", fontsize=12, transform=axes[1,3].transAxes)
axes[1, 3].set_title("Info Kamera")
axes[1, 3].axis("off")

plt.suptitle("Percobaan 07: Proyeksi 3D ke 2D", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "07_proyeksi_3d_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")
