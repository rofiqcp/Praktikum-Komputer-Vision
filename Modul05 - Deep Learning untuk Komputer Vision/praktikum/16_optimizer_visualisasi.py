"""
==========================================================================
PERCOBAAN 16: OPTIMIZER - VISUALISASI
==========================================================================
Program ini mengimplementasikan dan memvisualisasikan berbagai algoritma
optimasi (optimizer) yang digunakan dalam deep learning. Optimizer
bertanggung jawab untuk memperbarui parameter model berdasarkan gradien
yang dihitung selama backpropagation. Implementasi dilakukan dari nol
menggunakan NumPy untuk memahami mekanisme internal setiap optimizer.

Fungsi utama yang dipelajari:
- SGD (Stochastic Gradient Descent)  : Optimizer paling dasar
- SGD + Momentum                      : SGD dengan percepatan
- Adam (Adaptive Moment Estimation)   : Optimizer adaptif populer
- Operasi NumPy: np.zeros_like(), np.sqrt(), np.meshgrid()
- Matplotlib: plt.contour(), plt.contourf(), plt.plot()

Konsep yang dipelajari:
- Cara kerja SGD, Momentum, dan Adam
- Pengaruh learning rate terhadap konvergensi
- Perbandingan kecepatan konvergensi antar optimizer
- Learning rate scheduling: step decay, cosine annealing
- Visualisasi jalur optimasi pada loss surface
- Trade-off antara kecepatan dan stabilitas konvergensi
==========================================================================
"""

# Mengimpor NumPy untuk operasi komputasi numerik
import numpy as np

# Mengimpor os untuk operasi file dan path
import os

# Mengimpor matplotlib untuk visualisasi grafik
import matplotlib.pyplot as plt

# Mengimpor matplotlib colormap untuk pewarnaan
from matplotlib import cm

# Mendapatkan direktori script saat ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Menampilkan header percobaan
print("=" * 60)
print("PERCOBAAN 16: OPTIMIZER - VISUALISASI")
print("=" * 60)

# ============================================================
# 1. Penjelasan konsep optimizer
# ============================================================
print("\n--- 1. Konsep Optimizer dalam Deep Learning ---")

# Menjelaskan konsep optimizer
print("""
  Optimizer memperbarui parameter (bobot) model berdasarkan gradien:
    w_baru = w_lama - update

  Jenis-jenis optimizer:
  1. SGD        : update = lr * gradien
  2. Momentum   : update = lr * gradien + momentum * update_sebelum
  3. Adam       : update adaptif berdasarkan momen pertama & kedua

  | Optimizer  | Kelebihan              | Kekurangan            |
  |------------|------------------------|-----------------------|
  | SGD        | Sederhana, stabil      | Lambat konvergen      |
  | Momentum   | Lebih cepat, smooth    | 1 hyperparameter ekstra|
  | Adam       | Adaptif, cepat         | Kadang kurang generalize|
""")

# ============================================================
# 2. Mendefinisikan fungsi loss dan gradien
# ============================================================
print("\n--- 2. Mendefinisikan Fungsi Loss ---")


def rosenbrock(x, y, a=1, b=100):
    """
    Fungsi Rosenbrock: f(x,y) = (a-x)^2 + b*(y-x^2)^2
    Minimum global di (a, a^2) = (1, 1)
    """
    # Menghitung komponen pertama
    term1 = (a - x) ** 2

    # Menghitung komponen kedua
    term2 = b * (y - x ** 2) ** 2

    # Mengembalikan total loss
    return term1 + term2


def rosenbrock_grad(x, y, a=1, b=100):
    """
    Gradien fungsi Rosenbrock.
    df/dx = -2(a-x) + b*2*(y-x^2)*(-2x)
    df/dy = b*2*(y-x^2)
    """
    # Menghitung gradien terhadap x
    dx = -2.0 * (a - x) - 4.0 * b * x * (y - x ** 2)

    # Menghitung gradien terhadap y
    dy = 2.0 * b * (y - x ** 2)

    # Mengembalikan array gradien
    return np.array([dx, dy])


def quadratic(x, y):
    """
    Fungsi kuadratik sederhana: f(x,y) = x^2 + 5*y^2
    Minimum global di (0, 0). Elongated valley.
    """
    # Menghitung fungsi kuadratik dengan skala berbeda
    return x ** 2 + 5.0 * y ** 2


def quadratic_grad(x, y):
    """
    Gradien fungsi kuadratik.
    df/dx = 2x, df/dy = 10y
    """
    # Menghitung gradien terhadap x
    dx = 2.0 * x

    # Menghitung gradien terhadap y
    dy = 10.0 * y

    # Mengembalikan array gradien
    return np.array([dx, dy])


# Menampilkan informasi fungsi loss
print(f"  Fungsi Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2")
print(f"  Minimum global   : (1, 1)")
print(f"  Fungsi Kuadratik : f(x,y) = x^2 + 5*y^2")
print(f"  Minimum global   : (0, 0)")

# ============================================================
# 3. Implementasi SGD (Stochastic Gradient Descent)
# ============================================================
print("\n--- 3. Implementasi SGD ---")


def sgd_optimizer(grad_func, x0, y0, lr=0.001, n_iter=500):
    """
    Stochastic Gradient Descent (SGD).
    Update: w = w - lr * gradient
    """
    # Menginisialisasi posisi parameter
    x, y = x0, y0

    # Menyiapkan list untuk menyimpan jalur
    path_x = [x]
    path_y = [y]
    losses = []

    # Melakukan iterasi optimasi
    for i in range(n_iter):
        # Menghitung gradien di posisi saat ini
        grad = grad_func(x, y)

        # Memperbarui parameter: w = w - lr * grad
        x = x - lr * grad[0]
        y = y - lr * grad[1]

        # Menyimpan posisi baru
        path_x.append(x)
        path_y.append(y)

    # Mengembalikan jalur optimasi
    return np.array(path_x), np.array(path_y)


# Menampilkan formulasi SGD
print("  SGD: w_new = w_old - lr * gradient")
print("  Kelebihan: sederhana, mudah diimplementasi")
print("  Kekurangan: bisa lambat, sensitif terhadap learning rate")

# ============================================================
# 4. Implementasi SGD + Momentum
# ============================================================
print("\n--- 4. Implementasi SGD + Momentum ---")


def sgd_momentum_optimizer(grad_func, x0, y0, lr=0.001, momentum=0.9, n_iter=500):
    """
    SGD dengan Momentum.
    v = momentum * v - lr * gradient
    w = w + v
    """
    # Menginisialisasi posisi parameter
    x, y = x0, y0

    # Menginisialisasi velocity (kecepatan) menjadi nol
    vx, vy = 0.0, 0.0

    # Menyiapkan list untuk menyimpan jalur
    path_x = [x]
    path_y = [y]

    # Melakukan iterasi optimasi
    for i in range(n_iter):
        # Menghitung gradien di posisi saat ini
        grad = grad_func(x, y)

        # Memperbarui velocity: v = momentum * v - lr * grad
        vx = momentum * vx - lr * grad[0]
        vy = momentum * vy - lr * grad[1]

        # Memperbarui parameter: w = w + v
        x = x + vx
        y = y + vy

        # Menyimpan posisi baru
        path_x.append(x)
        path_y.append(y)

    # Mengembalikan jalur optimasi
    return np.array(path_x), np.array(path_y)


# Menampilkan formulasi Momentum
print("  Momentum: v = beta * v - lr * gradient")
print("            w = w + v")
print("  Kelebihan: lebih cepat, mengatasi local minima")
print("  Kekurangan: perlu tuning parameter momentum")

# ============================================================
# 5. Implementasi Adam Optimizer
# ============================================================
print("\n--- 5. Implementasi Adam Optimizer ---")


def adam_optimizer(grad_func, x0, y0, lr=0.01, beta1=0.9, beta2=0.999,
                   epsilon=1e-8, n_iter=500):
    """
    Adam (Adaptive Moment Estimation) Optimizer.
    m = beta1 * m + (1-beta1) * grad         (momen pertama)
    v = beta2 * v + (1-beta2) * grad^2       (momen kedua)
    m_hat = m / (1 - beta1^t)                (koreksi bias)
    v_hat = v / (1 - beta2^t)                (koreksi bias)
    w = w - lr * m_hat / (sqrt(v_hat) + eps)
    """
    # Menginisialisasi posisi parameter
    x, y = x0, y0

    # Menginisialisasi momen pertama (mean gradien)
    mx, my = 0.0, 0.0

    # Menginisialisasi momen kedua (mean kuadrat gradien)
    vx, vy = 0.0, 0.0

    # Menyiapkan list untuk menyimpan jalur
    path_x = [x]
    path_y = [y]

    # Melakukan iterasi optimasi
    for t in range(1, n_iter + 1):
        # Menghitung gradien di posisi saat ini
        grad = grad_func(x, y)

        # Memperbarui momen pertama: m = beta1 * m + (1-beta1) * grad
        mx = beta1 * mx + (1.0 - beta1) * grad[0]
        my = beta1 * my + (1.0 - beta1) * grad[1]

        # Memperbarui momen kedua: v = beta2 * v + (1-beta2) * grad^2
        vx = beta2 * vx + (1.0 - beta2) * grad[0] ** 2
        vy = beta2 * vy + (1.0 - beta2) * grad[1] ** 2

        # Mengoreksi bias momen pertama: m_hat = m / (1 - beta1^t)
        mx_hat = mx / (1.0 - beta1 ** t)
        my_hat = my / (1.0 - beta1 ** t)

        # Mengoreksi bias momen kedua: v_hat = v / (1 - beta2^t)
        vx_hat = vx / (1.0 - beta2 ** t)
        vy_hat = vy / (1.0 - beta2 ** t)

        # Memperbarui parameter: w = w - lr * m_hat / (sqrt(v_hat) + eps)
        x = x - lr * mx_hat / (np.sqrt(vx_hat) + epsilon)
        y = y - lr * my_hat / (np.sqrt(vy_hat) + epsilon)

        # Menyimpan posisi baru
        path_x.append(x)
        path_y.append(y)

    # Mengembalikan jalur optimasi
    return np.array(path_x), np.array(path_y)


# Menampilkan formulasi Adam
print("  Adam menggabungkan momentum dan adaptive learning rate")
print("  Hyperparameter: lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8")
print("  Kelebihan: adaptif, cepat konvergen, robust")

# ============================================================
# 6. Visualisasi jalur optimasi pada loss surface (Gambar 1)
# ============================================================
print("\n--- 6. Visualisasi Jalur Optimasi ---")

# Mendefinisikan titik awal yang sama untuk semua optimizer
x0_quad, y0_quad = -4.0, 3.0

# Menjalankan SGD pada fungsi kuadratik
sgd_x, sgd_y = sgd_optimizer(quadratic_grad, x0_quad, y0_quad,
                               lr=0.05, n_iter=100)

# Menjalankan SGD + Momentum pada fungsi kuadratik
mom_x, mom_y = sgd_momentum_optimizer(quadratic_grad, x0_quad, y0_quad,
                                        lr=0.05, momentum=0.9, n_iter=100)

# Menjalankan Adam pada fungsi kuadratik
adam_x, adam_y = adam_optimizer(quadratic_grad, x0_quad, y0_quad,
                                lr=0.5, n_iter=100)

# Membuat grid 2D untuk contour plot fungsi kuadratik
xg = np.linspace(-5, 5, 300)
yg = np.linspace(-4, 4, 300)

# Membuat meshgrid dari range parameter
XG, YG = np.meshgrid(xg, yg)

# Menghitung loss untuk setiap titik di grid
ZG = quadratic(XG, YG)

# Membuat figure dengan 2 subplot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Memberikan judul utama
fig.suptitle("Visualisasi Jalur Optimasi pada Loss Surface",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Contour plot dengan jalur optimizer ---
ax1 = axes[0]

# Menggambar filled contour plot
contour = ax1.contourf(XG, YG, ZG, levels=30, cmap='coolwarm', alpha=0.7)

# Menambahkan contour lines
ax1.contour(XG, YG, ZG, levels=15, colors='gray', alpha=0.3, linewidths=0.5)

# Menambahkan colorbar
plt.colorbar(contour, ax=ax1, label='Loss f(x,y) = x² + 5y²')

# Menggambar jalur SGD
ax1.plot(sgd_x, sgd_y, 'b.-', markersize=3, linewidth=1.5,
         label='SGD', alpha=0.8)

# Menggambar jalur Momentum
ax1.plot(mom_x, mom_y, 'g.-', markersize=3, linewidth=1.5,
         label='Momentum', alpha=0.8)

# Menggambar jalur Adam
ax1.plot(adam_x, adam_y, 'r.-', markersize=3, linewidth=1.5,
         label='Adam', alpha=0.8)

# Menandai titik awal
ax1.plot(x0_quad, y0_quad, 'k*', markersize=15, label='Start', zorder=5)

# Menandai titik minimum (0, 0)
ax1.plot(0, 0, 'w*', markersize=12, markeredgecolor='k', label='Minimum',
         zorder=5)

# Mengatur judul dan label
ax1.set_title("Jalur Optimasi: f(x,y) = x² + 5y²", fontsize=12,
              fontweight='bold')
ax1.set_xlabel("Parameter x")
ax1.set_ylabel("Parameter y")
ax1.legend(fontsize=9, loc='upper right')

# --- Subplot 2: Loss vs iterasi ---
ax2 = axes[1]

# Menghitung loss di setiap titik jalur SGD
sgd_losses = [quadratic(sgd_x[i], sgd_y[i]) for i in range(len(sgd_x))]

# Menghitung loss di setiap titik jalur Momentum
mom_losses = [quadratic(mom_x[i], mom_y[i]) for i in range(len(mom_x))]

# Menghitung loss di setiap titik jalur Adam
adam_losses = [quadratic(adam_x[i], adam_y[i]) for i in range(len(adam_x))]

# Menggambar kurva loss SGD
ax2.plot(range(len(sgd_losses)), sgd_losses, 'b-', linewidth=2, label='SGD')

# Menggambar kurva loss Momentum
ax2.plot(range(len(mom_losses)), mom_losses, 'g-', linewidth=2, label='Momentum')

# Menggambar kurva loss Adam
ax2.plot(range(len(adam_losses)), adam_losses, 'r-', linewidth=2, label='Adam')

# Mengatur skala y ke logaritmik untuk melihat konvergensi lebih jelas
ax2.set_yscale('log')

# Mengatur judul dan label
ax2.set_title("Loss vs Iterasi (skala log)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Iterasi")
ax2.set_ylabel("Loss (log scale)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_1 = os.path.join(OUTPUT_DIR, "16_optimizer_path.png")

# Menyimpan figure
plt.savefig(output_path_1, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_1}")

# ============================================================
# 7. Perbandingan konvergensi dan efek learning rate (Gambar 2)
# ============================================================
print("\n--- 7. Perbandingan Konvergensi dan Efek Learning Rate ---")

# Membuat figure dengan 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Memberikan judul utama
fig.suptitle("Perbandingan Konvergensi dan Efek Learning Rate",
             fontsize=16, fontweight='bold')

# --- Subplot 1: Efek learning rate pada SGD ---
ax1 = axes[0, 0]

# Mendefinisikan beberapa learning rate untuk diuji
learning_rates = [0.001, 0.01, 0.05, 0.1]

# Menguji setiap learning rate
for lr in learning_rates:
    # Menjalankan SGD dengan learning rate tertentu
    sx, sy = sgd_optimizer(quadratic_grad, x0_quad, y0_quad, lr=lr, n_iter=100)

    # Menghitung loss di setiap iterasi
    losses = [quadratic(sx[i], sy[i]) for i in range(len(sx))]

    # Menggambar kurva loss
    ax1.plot(range(len(losses)), losses, linewidth=1.5, label=f'lr={lr}')

# Mengatur skala y ke logaritmik
ax1.set_yscale('log')

# Mengatur judul dan label
ax1.set_title("SGD: Efek Learning Rate", fontsize=11, fontweight='bold')
ax1.set_xlabel("Iterasi")
ax1.set_ylabel("Loss (log)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Efek learning rate pada Adam ---
ax2 = axes[0, 1]

# Mendefinisikan learning rates untuk Adam
adam_lrs = [0.01, 0.05, 0.1, 0.5]

# Menguji setiap learning rate
for lr in adam_lrs:
    # Menjalankan Adam dengan learning rate tertentu
    ax_path, ay_path = adam_optimizer(quadratic_grad, x0_quad, y0_quad,
                                      lr=lr, n_iter=100)

    # Menghitung loss di setiap iterasi
    losses = [quadratic(ax_path[i], ay_path[i]) for i in range(len(ax_path))]

    # Menggambar kurva loss
    ax2.plot(range(len(losses)), losses, linewidth=1.5, label=f'lr={lr}')

# Mengatur skala y ke logaritmik
ax2.set_yscale('log')

# Mengatur judul dan label
ax2.set_title("Adam: Efek Learning Rate", fontsize=11, fontweight='bold')
ax2.set_xlabel("Iterasi")
ax2.set_ylabel("Loss (log)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Efek momentum pada SGD+Momentum ---
ax3 = axes[1, 0]

# Mendefinisikan beberapa nilai momentum
momentum_values = [0.0, 0.5, 0.9, 0.99]

# Menguji setiap nilai momentum
for mom in momentum_values:
    # Menjalankan SGD+Momentum
    mx_path, my_path = sgd_momentum_optimizer(quadratic_grad, x0_quad, y0_quad,
                                               lr=0.01, momentum=mom, n_iter=200)

    # Menghitung loss di setiap iterasi
    losses = [quadratic(mx_path[i], my_path[i]) for i in range(len(mx_path))]

    # Menggambar kurva loss
    label_str = f'β={mom}' if mom > 0 else 'SGD murni'
    ax3.plot(range(len(losses)), losses, linewidth=1.5, label=label_str)

# Mengatur skala y ke logaritmik
ax3.set_yscale('log')

# Mengatur judul dan label
ax3.set_title("Efek Momentum (β)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Iterasi")
ax3.set_ylabel("Loss (log)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# --- Subplot 4: Perbandingan final distances ---
ax4 = axes[1, 1]

# Menghitung jarak akhir ke minimum untuk setiap optimizer
optimizer_names = ['SGD\n(lr=0.05)', 'Momentum\n(lr=0.05,β=0.9)',
                   'Adam\n(lr=0.5)']

# Menghitung jarak akhir SGD
dist_sgd = np.sqrt(sgd_x[-1] ** 2 + sgd_y[-1] ** 2)

# Menghitung jarak akhir Momentum
dist_mom = np.sqrt(mom_x[-1] ** 2 + mom_y[-1] ** 2)

# Menghitung jarak akhir Adam
dist_adam = np.sqrt(adam_x[-1] ** 2 + adam_y[-1] ** 2)

# Mengumpulkan jarak
distances = [dist_sgd, dist_mom, dist_adam]

# Menggambar bar chart jarak akhir
colors = ['steelblue', 'forestgreen', 'indianred']
bars = ax4.bar(optimizer_names, distances, color=colors)

# Menambahkan label nilai di atas bar
for bar, dist in zip(bars, distances):
    ax4.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
             f'{dist:.4f}', ha='center', va='bottom', fontsize=10)

# Mengatur judul dan label
ax4.set_title("Jarak Akhir ke Minimum (0,0)", fontsize=11, fontweight='bold')
ax4.set_ylabel("Jarak Euclidean")
ax4.grid(True, alpha=0.3, axis='y')

# Menampilkan informasi konvergensi
print(f"  SGD final distance     : {dist_sgd:.6f}")
print(f"  Momentum final distance: {dist_mom:.6f}")
print(f"  Adam final distance    : {dist_adam:.6f}")

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_2 = os.path.join(OUTPUT_DIR, "16_optimizer_convergence.png")

# Menyimpan figure
plt.savefig(output_path_2, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_2}")

# ============================================================
# 8. Learning Rate Scheduling (Gambar 3)
# ============================================================
print("\n--- 8. Learning Rate Scheduling ---")

# Mendefinisikan jumlah total epoch
total_epochs = 100

# Mendefinisikan learning rate awal
lr_awal = 0.1

# Array epoch
epochs = np.arange(total_epochs)

# --- Implementasi Step Decay ---
def step_decay(epoch, lr_init=0.1, drop_factor=0.5, drop_every=20):
    """
    Step Decay: menurunkan LR sebesar faktor tertentu setiap N epoch.
    """
    # Menghitung berapa kali sudah diturunkan
    n_drops = epoch // drop_every

    # Menghitung learning rate baru
    lr = lr_init * (drop_factor ** n_drops)

    # Mengembalikan learning rate
    return lr


# --- Implementasi Cosine Annealing ---
def cosine_annealing(epoch, lr_init=0.1, total_epochs=100, lr_min=0.001):
    """
    Cosine Annealing: menurunkan LR mengikuti kurva cosinus.
    """
    # Menghitung learning rate dengan formula cosine
    lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * epoch / total_epochs))

    # Mengembalikan learning rate
    return lr


# --- Implementasi Exponential Decay ---
def exponential_decay(epoch, lr_init=0.1, decay_rate=0.95):
    """
    Exponential Decay: menurunkan LR secara eksponensial.
    """
    # Menghitung learning rate baru
    lr = lr_init * (decay_rate ** epoch)

    # Mengembalikan learning rate
    return lr


# --- Implementasi Warmup + Decay ---
def warmup_cosine(epoch, lr_init=0.1, warmup_epochs=10, total_epochs=100):
    """
    Warmup lalu Cosine Annealing: LR naik linear selama warmup,
    lalu turun mengikuti cosine.
    """
    if epoch < warmup_epochs:
        # Fase warmup: meningkatkan LR secara linear
        lr = lr_init * (epoch + 1) / warmup_epochs
    else:
        # Fase cosine annealing setelah warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr_init * 0.5 * (1 + np.cos(np.pi * progress))

    # Mengembalikan learning rate
    return lr


# Menghitung LR untuk setiap schedule
lr_step = [step_decay(e, lr_awal) for e in epochs]
lr_cosine = [cosine_annealing(e, lr_awal, total_epochs) for e in epochs]
lr_exp = [exponential_decay(e, lr_awal) for e in epochs]
lr_warmup = [warmup_cosine(e, lr_awal, 10, total_epochs) for e in epochs]

# Menghitung konstan LR sebagai baseline
lr_const = [lr_awal] * total_epochs

# Membuat figure untuk LR scheduling
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Memberikan judul utama
fig.suptitle("Learning Rate Scheduling", fontsize=16, fontweight='bold')

# --- Subplot 1: Semua schedule dalam satu plot ---
ax1 = axes[0, 0]

# Menggambar LR konstan
ax1.plot(epochs, lr_const, 'k--', linewidth=1.5, label='Konstan', alpha=0.5)

# Menggambar step decay
ax1.plot(epochs, lr_step, 'b-', linewidth=2, label='Step Decay')

# Menggambar cosine annealing
ax1.plot(epochs, lr_cosine, 'r-', linewidth=2, label='Cosine Annealing')

# Menggambar exponential decay
ax1.plot(epochs, lr_exp, 'g-', linewidth=2, label='Exponential Decay')

# Menggambar warmup + cosine
ax1.plot(epochs, lr_warmup, 'm-', linewidth=2, label='Warmup + Cosine')

# Mengatur judul dan label
ax1.set_title("Perbandingan LR Schedule", fontsize=11, fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Learning Rate")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Efek schedule pada konvergensi ---
ax2 = axes[0, 1]

# Menjalankan optimasi dengan berbagai LR schedule
schedule_names = ['Konstan', 'Step Decay', 'Cosine', 'Warmup+Cosine']
schedule_funcs = [
    lambda e: lr_awal,
    lambda e: step_decay(e, lr_awal),
    lambda e: cosine_annealing(e, lr_awal, total_epochs),
    lambda e: warmup_cosine(e, lr_awal, 10, total_epochs),
]
schedule_colors = ['gray', 'blue', 'red', 'purple']

# Menguji setiap schedule
for name, schedule_fn, color in zip(schedule_names, schedule_funcs, schedule_colors):
    # Menginisialisasi parameter
    x, y = x0_quad, y0_quad
    losses_sched = [quadratic(x, y)]

    # Melakukan optimasi dengan LR schedule
    for e in range(total_epochs):
        # Mendapatkan learning rate saat ini
        cur_lr = schedule_fn(e)

        # Menghitung gradien
        grad = quadratic_grad(x, y)

        # Memperbarui parameter
        x = x - cur_lr * grad[0]
        y = y - cur_lr * grad[1]

        # Menyimpan loss
        losses_sched.append(quadratic(x, y))

    # Menggambar kurva loss
    ax2.plot(range(len(losses_sched)), losses_sched, color=color,
             linewidth=1.5, label=name)

# Mengatur skala y ke logaritmik
ax2.set_yscale('log')

# Mengatur judul dan label
ax2.set_title("Efek Schedule pada Konvergensi", fontsize=11, fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (log)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Detail Step Decay ---
ax3 = axes[1, 0]

# Menghitung step decay dengan parameter berbeda
sd_configs = [
    (0.5, 20, 'drop=0.5, every=20'),
    (0.5, 10, 'drop=0.5, every=10'),
    (0.1, 30, 'drop=0.1, every=30'),
    (0.3, 25, 'drop=0.3, every=25'),
]

# Menguji setiap konfigurasi step decay
for drop_f, drop_e, label in sd_configs:
    # Menghitung LR untuk setiap epoch
    lr_vals = [step_decay(e, lr_awal, drop_f, drop_e) for e in epochs]

    # Menggambar kurva LR
    ax3.plot(epochs, lr_vals, linewidth=2, label=label)

# Mengatur judul dan label
ax3.set_title("Variasi Step Decay", fontsize=11, fontweight='bold')
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- Subplot 4: Tabel ringkasan ---
ax4 = axes[1, 1]

# Menyembunyikan axes
ax4.axis('off')

# Membuat data tabel ringkasan
tabel_data = [
    ['Konstan', 'Tidak berubah', 'Baseline'],
    ['Step Decay', 'Turun per N epoch', 'Paling umum'],
    ['Cosine', 'Mengikuti cosinus', 'Smooth decay'],
    ['Exp. Decay', 'Turun eksponensial', 'Cepat turun'],
    ['Warmup+Cos', 'Naik lalu cosine', 'Training stabil'],
]

# Mendefinisikan header tabel
kolom_header = ['Schedule', 'Pola', 'Karakteristik']

# Menggambar tabel
tabel = ax4.table(cellText=tabel_data, colLabels=kolom_header,
                  loc='center', cellLoc='center')

# Mengatur ukuran font
tabel.auto_set_font_size(False)
tabel.set_fontsize(10)

# Mengatur skala tabel
tabel.scale(1.0, 1.8)

# Mewarnai header
for j in range(len(kolom_header)):
    tabel[0, j].set_facecolor('#1976D2')
    tabel[0, j].set_text_props(color='white', fontweight='bold')

# Mengatur judul
ax4.set_title("Ringkasan LR Schedule", fontsize=11, fontweight='bold', pad=20)

# Mengatur layout
plt.tight_layout()

# Mendefinisikan path output
output_path_3 = os.path.join(OUTPUT_DIR, "16_learning_rate_schedule.png")

# Menyimpan figure
plt.savefig(output_path_3, dpi=150, bbox_inches='tight')

# Menutup figure
plt.close()

# Menampilkan pesan konfirmasi
print(f"  [SAVED] {output_path_3}")

# ============================================================
# 9. Analisis numerik tambahan
# ============================================================
print("\n--- 9. Analisis Numerik ---")

# Menampilkan perbandingan konvergensi
print("\n  Perbandingan posisi akhir setelah 100 iterasi:")
print(f"  {'Optimizer':<20} {'x_final':<12} {'y_final':<12} {'Loss_final':<12}")
print(f"  {'-'*56}")

# Menghitung dan menampilkan untuk SGD
loss_sgd = quadratic(sgd_x[-1], sgd_y[-1])
print(f"  {'SGD':<20} {sgd_x[-1]:<12.6f} {sgd_y[-1]:<12.6f} {loss_sgd:<12.6f}")

# Menghitung dan menampilkan untuk Momentum
loss_mom = quadratic(mom_x[-1], mom_y[-1])
print(f"  {'Momentum':<20} {mom_x[-1]:<12.6f} {mom_y[-1]:<12.6f} {loss_mom:<12.6f}")

# Menghitung dan menampilkan untuk Adam
loss_adam_final = quadratic(adam_x[-1], adam_y[-1])
print(f"  {'Adam':<20} {adam_x[-1]:<12.6f} {adam_y[-1]:<12.6f} {loss_adam_final:<12.6f}")

# ============================================================
# 10. Ringkasan
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN PERCOBAAN 16")
print("=" * 60)
print(f"""
Konsep yang telah dipelajari:
1. SGD: optimizer paling sederhana, w = w - lr * grad
2. Momentum: menambahkan "inersia", v = beta*v - lr*grad
3. Adam: kombinasi momentum + adaptive LR, paling populer
4. Learning rate besar -> cepat tapi bisa diverge
5. Learning rate kecil -> stabil tapi lambat
6. LR scheduling membantu konvergensi yang lebih baik
7. Cosine annealing memberikan penurunan LR yang smooth
8. Warmup membantu stabilisasi awal training

Hasil Konvergensi (100 iterasi):
- SGD      : loss = {loss_sgd:.6f}
- Momentum : loss = {loss_mom:.6f}
- Adam     : loss = {loss_adam_final:.6f}

Output disimpan di folder: output/
- 16_optimizer_path.png          : Jalur optimasi pada loss surface
- 16_optimizer_convergence.png   : Perbandingan konvergensi dan LR
- 16_learning_rate_schedule.png  : Visualisasi LR scheduling
""")
