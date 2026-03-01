# TUGAS VIDEO MODUL 10: COMPUTATIONAL PHOTOGRAPHY

---

## Deskripsi Tugas
Buat video laporan yang mendemonstrasikan seluruh materi, 20 percobaan, dan project computational photography. Tunjukkan perbandingan visual yang jelas.

---

## Struktur Video

### 1. Pembukaan (Maks 2 menit)
- Perkenalan: Nama, NIM, kelas, modul.
- Overview topik Computational Photography.

### 2. Penjelasan Materi (10–15 menit)
- HDR: pipeline, CRF, tone mapping operators.
- Exposure fusion: Mertens quality measures.
- Denoising: Gaussian, bilateral, NLM — perbedaan prinsip.
- Inpainting: NS, Telea.
- Super resolution: interpolasi vs DNN.
- Enhancement pipeline: urutan optimal.
- Style transfer: content loss + style loss.
- **Wajib**: Diagram HDR pipeline + denoising comparison.

### 3. Demo 20 Percobaan (40–60 menit)

| No | Percobaan | Poin Penting |
|----|-----------|--------------|
| 1 | HDR Imaging | 3 exposures → HDR → 3 tone mappers |
| 2 | Tone Mapping Reinhard | Parameter gamma, light adapt |
| 3 | Tone Mapping Drago | Saturation, bias parameter |
| 4 | Exposure Fusion | Mertens vs HDR comparison |
| 5 | Denoising Gaussian | Kernel size, sigma efek |
| 6 | Denoising Bilateral | Edge-preserving, sigmaColor |
| 7 | Denoising NLM | Patch size, search window |
| 8 | Inpainting NS | Navier-Stokes, mask painting |
| 9 | Inpainting Telea | Fast marching, NS vs Telea |
| 10 | Super Resolution | Interpolasi bilinear, bicubic, Lanczos |
| 11 | CLAHE Enhancement | clipLimit, tileGridSize |
| 12 | Unsharp Mask | Sigma, amount, threshold |
| 13 | White Balance | Gray world, white patch |
| 14 | Synthetic Bokeh | Depth-based blur, DOF |
| 15 | Color Enhancement | Saturation, vibrance |
| 16 | Enhancement Pipeline | Full chain: denoise→WB→CLAHE→sharpen |
| 17 | Pencil Sketch | cv2.pencilSketch(), edge |
| 18 | Cartoon Effect | Bilateral + edge overlay |
| 19 | HDR Single Image | Pseudo-HDR dari 1 foto |
| 20 | Style Transfer Manual | Gram matrix, DNN transfer |

- Tunjukkan foto asli yang Anda ambil sendiri (selfie/landscape).
- Before/after comparison harus jelas dan close-up.

### 4. Demo Project (10–15 menit)
- Demo project soal cerita.
- Tunjukkan semua fitur + visual output.

### 5. Analisis dan Penutup (5 menit)
- Rangkuman perbandingan metode.
- Kapan menggunakan teknik mana.
- Kesimpulan.

---

## Ketentuan Teknis

| Aspek | Ketentuan |
|-------|-----------|
| Durasi | 75–100 menit |
| Resolusi | Minimal 1080p |
| Recording | Screen recording + webcam |
| Audio | Narasi jelas |
| Platform | YouTube (Unlisted) atau Google Drive |
| Format | MP4 |

---

## Rubrik Penilaian Video

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Pembukaan | 5% | Profesional |
| Penjelasan Materi | 15% | Akurat, diagram |
| Demo 20 Percobaan | 40% | Semua berjalan, comparison jelas |
| Demo Project | 20% | Fitur lengkap |
| Analisis & Kesimpulan | 10% | Kritis |
| Kualitas Video | 10% | Resolusi, audio |

### Bonus & Penalti
| Item | Nilai |
|------|-------|
| Gunakan foto sendiri (bukan download) untuk semua percobaan | +5 |
| Tabel PSNR/SSIM lengkap semua metode | +5 |
| Video < 45 menit | -10 |
| Tidak ada webcam | -5 |
| Audio tidak jelas | -5 |
| Percobaan error tanpa penjelasan | -5 per percobaan |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: Link YouTube (Unlisted) atau Google Drive.
- **Naming**: `[NIM]_[Nama]_Video_Modul10`
