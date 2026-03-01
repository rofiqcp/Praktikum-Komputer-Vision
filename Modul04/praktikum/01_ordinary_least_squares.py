"""
==========================================================================
PERCOBAAN 01: ORDINARY LEAST SQUARES (OLS)
==========================================================================
OLS meminimalkan jumlah kuadrat residual antara data observasi dan
model prediksi. Ini adalah dasar dari semua teknik model fitting.

Rumus: min ||Ax - b||^2
Solusi: x = (A^T A)^(-1) A^T b

Fungsi:
- np.linalg.lstsq(A, b) → solusi least squares
- np.polyfit(x, y, deg) → fitting polinomial
- np.polyval(p, x) → evaluasi polinomial
- cv2.fitLine(pts, distType, ...) → fit garis ke titik
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PERCOBAAN 01: ORDINARY LEAST SQUARES (OLS)")
print("=" * 60)

np.random.seed(42)

# ============================================================
# 1. Fitting Garis Lurus (Linear Regression)
# ============================================================
print("\n--- 1. Linear Fitting (y = ax + b) ---")

# Buat data sintetis: y = 2x + 10 + noise
N = 100
x_data = np.linspace(0, 10, N)
y_true = 2 * x_data + 10
noise = np.random.randn(N) * 3
y_data = y_true + noise

# Susun matrix A = [x, 1]
A = np.vstack([x_data, np.ones(N)]).T
# Solusi least squares: (A^T A)^(-1) A^T b
params_manual = np.linalg.inv(A.T @ A) @ A.T @ y_data
print(f"  Manual: a={params_manual[0]:.4f}, b={params_manual[1]:.4f}")

# Menggunakan np.linalg.lstsq (lebih stabil numerik)
params_lstsq, residuals, rank, sv = np.linalg.lstsq(A, y_data, rcond=None)
print(f"  lstsq:  a={params_lstsq[0]:.4f}, b={params_lstsq[1]:.4f}")

# Menggunakan np.polyfit
coeffs = np.polyfit(x_data, y_data, 1)
print(f"  polyfit: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}")

# Hitung R-squared
y_pred = params_lstsq[0] * x_data + params_lstsq[1]
ss_res = np.sum((y_data - y_pred) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"  R² = {r_squared:.4f}")

# ============================================================
# 2. Fitting Polinomial (Quadratic, Cubic)
# ============================================================
print("\n--- 2. Polynomial Fitting ---")

# Data kuadrat: y = 0.5x^2 - 3x + 7
y_quad = 0.5 * x_data**2 - 3 * x_data + 7 + np.random.randn(N) * 2

# Fit derajat 1, 2, 3
poly_results = {}
for deg in [1, 2, 3]:
    p = np.polyfit(x_data, y_quad, deg)
    y_fit = np.polyval(p, x_data)
    err = np.mean((y_quad - y_fit)**2)
    poly_results[deg] = (p, y_fit, err)
    print(f"  Derajat {deg}: MSE = {err:.4f}")

# ============================================================
# 3. Fitting Garis dari Gambar (cv2.fitLine)
# ============================================================
print("\n--- 3. cv2.fitLine ---")

# Baca gambar titik-titik
img_pts = cv2.imread(os.path.join(IMAGE_DIR, "garis_noise.png"))
if img_pts is not None:
    gray = cv2.cvtColor(img_pts, cv2.COLOR_BGR2GRAY)
    # Deteksi titik-titik (threshold pada titik merah)
    hsv = cv2.cvtColor(img_pts, cv2.COLOR_BGR2HSV)
    # Ambil semua titik yang cukup gelap (bukan putih)
    mask = gray < 200
    # Ambil koordinat titik
    pts = np.column_stack(np.where(mask))  # (y, x)
    pts_xy = pts[:, ::-1].astype(np.float32)  # (x, y)

    if len(pts_xy) > 10:
        # cv2.fitLine: distType, param, reps, aeps
        # cv2.DIST_L2 = least squares
        line = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()
        print(f"  Vektor arah: ({vx:.4f}, {vy:.4f})")
        print(f"  Titik pada garis: ({x0:.1f}, {y0:.1f})")
        # Gambar garis pada gambar
        img_result = img_pts.copy()
        t = 500
        pt1 = (int(x0 - t * vx), int(y0 - t * vy))
        pt2 = (int(x0 + t * vx), int(y0 + t * vy))
        cv2.line(img_result, pt1, pt2, (0, 255, 0), 2)
else:
    img_result = None
    print("  [SKIP] Gambar tidak ditemukan")

# ============================================================
# 4. Perbandingan Distance Type pada fitLine
# ============================================================
print("\n--- 4. Perbandingan distType ---")

# Buat titik dengan outlier
pts_outlier = []
for i in range(80):
    x = np.random.uniform(50, 450)
    y = 0.5 * x + 100 + np.random.randn() * 5
    pts_outlier.append([x, y])
# Tambah outlier jauh
for i in range(20):
    pts_outlier.append([np.random.uniform(50, 450), np.random.uniform(50, 450)])
pts_arr = np.array(pts_outlier, dtype=np.float32)

dist_types = {
    "DIST_L2": cv2.DIST_L2,
    "DIST_L1": cv2.DIST_L1,
    "DIST_L12": cv2.DIST_L12,
    "DIST_HUBER": cv2.DIST_HUBER,
}

fit_results = {}
for name, dtype in dist_types.items():
    line = cv2.fitLine(pts_arr, dtype, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    # Konversi ke y = mx + c
    slope = vy / vx if abs(vx) > 1e-6 else float('inf')
    fit_results[name] = (vx, vy, x0, y0, slope)
    print(f"  {name}: slope={slope:.4f}")

# ============================================================
# 5. Visualisasi
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Linear fit
axes[0, 0].scatter(x_data, y_data, s=10, alpha=0.5, label="Data")
axes[0, 0].plot(x_data, y_pred, 'r-', linewidth=2, label=f"OLS: y={params_lstsq[0]:.2f}x+{params_lstsq[1]:.2f}")
axes[0, 0].plot(x_data, y_true, 'g--', linewidth=1, label="True")
axes[0, 0].set_title(f"Linear Fit (R²={r_squared:.4f})")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Polynomial fit
axes[0, 1].scatter(x_data, y_quad, s=10, alpha=0.5, label="Data")
for deg, (p, y_f, err) in poly_results.items():
    axes[0, 1].plot(x_data, y_f, linewidth=2, label=f"deg={deg}, MSE={err:.2f}")
axes[0, 1].set_title("Polynomial Fitting")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: fitLine result
if img_result is not None:
    axes[1, 0].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("cv2.fitLine (DIST_L2)")
axes[1, 0].axis("off")

# Plot 4: Distance type comparison
axes[1, 1].scatter(pts_arr[:, 0], pts_arr[:, 1], s=10, alpha=0.5)
colors_line = ['r', 'g', 'b', 'm']
for (name, (vx, vy, x0, y0, slope)), col in zip(fit_results.items(), colors_line):
    xs = np.linspace(50, 450, 100)
    ys = y0 + (xs - x0) * vy / vx
    axes[1, 1].plot(xs, ys, col, linewidth=2, label=name)
axes[1, 1].set_title("Perbandingan distType")
axes[1, 1].legend()
axes[1, 1].set_xlim(0, 500); axes[1, 1].set_ylim(0, 500)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("Percobaan 01: Ordinary Least Squares", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "01_ols_hasil.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n[OUTPUT] {path}")

# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 01")
print("=" * 60)
print("""
1. OLS meminimalkan ||Ax - b||² → solusi: x = (AᵀA)⁻¹Aᵀb
2. np.linalg.lstsq() lebih stabil dari inverse langsung
3. np.polyfit() untuk fitting polinomial (derajat n)
4. cv2.fitLine() untuk fitting garis ke kumpulan titik 2D
5. DIST_L2 = least squares, DIST_L1 = median, DIST_HUBER = robust
6. R² mendekati 1 → model fit baik
7. Derajat polinomial terlalu tinggi → overfitting
""")
