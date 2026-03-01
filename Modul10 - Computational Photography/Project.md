# PROJECT MODUL 10: COMPUTATIONAL PHOTOGRAPHY

---

## Deskripsi Umum
Project mengintegrasikan konsep computational photography: HDR, denoising, inpainting, super resolution, style transfer, dan enhancement. Pilih minimal 1 soal cerita.

---

## Daftar Improvisasi Percobaan (15 Pengembangan)

1. **Auto-HDR dari Video** — Extract frames dengan exposure berbeda dari video → merge HDR.
2. **Night Photo Enhancer** — Pipeline khusus foto malam: denoise + enhance + brighten.
3. **Batch Photo Enhancer** — Enhancement pipeline otomatis untuk folder penuh foto.
4. **Before/After Slider** — Interactive slider untuk membandingkan gambar sebelum/sesudah.
5. **AI Background Remover + Replace** — Segment foreground → replace background → blend.
6. **Photo Restoration Tool** — Gabungkan denoising + inpainting + enhancement untuk foto lama.
7. **Text/Watermark Remover** — Otomatis deteksi + inpaint watermark.
8. **HDR Video Creator** — Tone mapping per frame → HDR video.
9. **Dual Camera Bokeh** — Gunakan 2 foto (fokus berbeda) → bokeh realistis.
10. **Custom Style Transfer Model** — Train model style transfer pada lukisan sendiri (jika GPU ada).
11. **Photo Colorization** — Warnai foto hitam-putih (DNN-based).
12. **Image Forensics** — Deteksi area yang di-inpaint/manipulasi.
13. **Dynamic Range Analyzer** — Visualisasikan dynamic range gambar sebagai histogram + zones.
14. **EXIF-aware Enhancement** — Baca EXIF → pilih enhancement pipeline sesuai kondisi.
15. **Artistic Filter Gallery** — 10+ filter combination (vintage, noir, pop art, dll).

---

## Soal Cerita Project (Pilih Minimal 1)

### Soal 1: Aplikasi Edit Foto Sederhana
Developer ingin membuat tools edit foto all-in-one. Buatlah: (a) auto enhance (brightness, contrast, color), (b) denoising (pilih level), (c) HDR dari single image (pseudo-HDR), (d) inpainting tool (gambar mask → hapus objek), (e) super resolution 2×/4×, (f) style transfer (pilih style), (g) before/after comparison.

### Soal 2: Sistem Restorasi Foto Lama
Perpustakaan ingin merestorasi koleksi foto lama. Buatlah: (a) scan foto lama (atau simulasikan degradasi), (b) denoising untuk menghilangkan grain, (c) inpainting untuk menghilangkan goresan/noda, (d) contrast enhancement (CLAHE), (e) super resolution untuk memperbesar detail, (f) colorization (opsional — DNN), (g) batch processing.

### Soal 3: Sistem HDR untuk Real Estate Photography
Agen properti memerlukan foto interior HDR berkualitas. Buatlah: (a) capture bracketed (atau gunakan sample), (b) HDR merge + multiple tone mapping options, (c) exposure fusion sebagai alternatif, (d) auto-crop + straighten, (e) enhancement pipeline (white balance, sharpen), (f) batch mode per ruangan, (g) export high-quality JPEG.

### Soal 4: Tool Penghapus Objek Otomatis
Content creator memerlukan tool menghapus objek tidak diinginkan. Buatlah: (a) deteksi objek (YOLO) → auto-generate mask, (b) user bisa pilih objek mana yang mau dihapus, (c) inpainting area objek, (d) multi-pass inpainting untuk area besar, (e) comparison before/after, (f) support drag-draw mask manual, (g) undo/redo.

### Soal 5: Night Photography Enhancement Suite
Fotografer sering mengambil foto malam yang noisy dan gelap. Buatlah: (a) multi-frame averaging dari burst photos, (b) NLM denoising, (c) brightness boost + contrast, (d) super resolution, (e) synthetic long exposure (light trails dari video malam), (f) comparison matrix (setiap step), (g) parameter tuning UI.

### Soal 6: Social Media Filter App
Startup media sosial ingin membangun filter foto. Buatlah: (a) 5 filter preset (vintage, B&W, warm, cool, dramatic), (b) style transfer (3 artistic styles), (c) bokeh effect dengan depth estimation, (d) beauty mode (smooth skin, brighten), (e) frame/border options, (f) text overlay, (g) export optimized (compressed JPEG).

### Soal 7: Panoramic HDR Generator
Fotografer landscape ingin tool panorama HDR. Buatlah: (a) input: multi-position × multi-exposure, (b) HDR merge per posisi, (c) tone mapping (user-selectable), (d) stitch panorama, (e) exposure compensation antar posisi, (f) crop + finalize, (g) export high-res.

### Soal 8: Dokumen/Whiteboard Enhancer
Mahasiswa ingin meningkatkan kualitas foto catatan. Buatlah: (a) perspective correction, (b) binarisasi adaptif, (c) denoising, (d) contrast enhancement, (e) inpainting (hapus tangan/bayangan), (f) super resolution untuk tulisan kecil, (g) export PDF multi-halaman.

### Soal 9: Microscopy Image Enhancer
Lab biologi memerlukan tool peningkatan gambar mikroskop. Buatlah: (a) multi-frame averaging (stack focus), (b) denoising (NLM), (c) contrast enhancement (CLAHE), (d) sharpening, (e) false coloring, (f) measurement overlay (scale bar), (g) comparison view.

### Soal 10: AI Art Generator
Seniman digital ingin tool kreasi seni berbasis foto. Buatlah: (a) style transfer dari 5 lukisan terkenal, (b) combinasi style (multi-style blending), (c) artistic edge detection (pencil sketch, oil painting effect), (d) poster effect (color quantization), (e) halftone effect, (f) pixel art conversion, (g) gallery mode.

---

## Rubrik Penilaian Project

| Komponen | Bobot | Keterangan |
|----------|-------|------------|
| Fungsionalitas | 35% | Semua fitur berjalan |
| Integrasi Percobaan | 20% | Menggunakan konsep ≥10 percobaan |
| Kualitas Kode | 15% | Clean, modular |
| Dokumentasi | 15% | README, screenshot, comparison |
| Kreativitas | 15% | Fitur tambahan, UI, visual output |

---

## Format Pengumpulan
- **Deadline**: 1 minggu setelah modul selesai.
- **Format**: ZIP — `NIM_Nama_Project10.zip`
- **Isi**: Source code, sample images, model weights (atau link), README.md.
