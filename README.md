# 🫆 Fingerprint Matcher — UAS Computer Vision

Sistem pencocokan & identifikasi sidik jari berbasis **OpenCV** dengan
dashboard web modern. Dibangun dari materi labsheet LS 1–7:
grayscale → **CLAHE** → filter → thresholding → morfologi → **ORB matching**.

> Dua mode: **Verifikasi** ("apakah jari ini milik X?") dan
> **Identifikasi** ("jari ini milik siapa?" — scan database pengguna).

---

## ⚡ Cara Run (paling cepat)

```bash
# 1. Clone repo
git clone git@github.com:rayatajemundergrounddigigtal/comvis_fingerprint.git
cd comvis_fingerprint

# 2. Install dependensi (sekali saja)
pip install -r requirements.txt

# 3. Buat sampel latihan + database pengguna (Andi, Budi, Citra)
python3 generate_samples.py

# 4. Buka DASHBOARD WEB 🌐
streamlit run app.py
```

Browser terbuka otomatis di **http://localhost:8501**:

| Tab | Isi |
|---|---|
| 🔐 **Verifikasi** | Upload 2 gambar jari → otomatis: ✅ COCOK / ❌ DITOLAK + skor + panel tahapan |
| 🔍 **Identifikasi** | Upload **1 gambar** → klik "CARI PEMILIKNYA" → nama pemilik + ranking kandidat |
| 📚 **Tentang** | Peta materi labsheet + cara kerja + analogi penjelasan |
| Sidebar | Slider ambang keputusan, CLAHE, grid tile, unsharp — hasil update live |

Semua proses berjalan **lokal** (localhost) — tidak ada data terkirim ke internet.

---

## 🔁 Alternatif tanpa browser: GUI OpenCV

```bash
python3 main.py                                       # demo default (A vs A-rotasi)
python3 main.py samples/jari_A.png samples/jari_B.png # kasus ditolak
python3 main.py --db database/ samples/jari_A_rotasi.png  # identifikasi via CLI
```

Kontrol GUI: trackbar (Threshold/ClipLimit/Unsharp) + tombol
`o` query · `r` referensi · `i` identifikasi · `v` verifikasi · `s` simpan · `q` keluar.

---

## 🧪 Uji akurasi (untuk tabel laporan)

```bash
python3 evaluate.py --synthetic              # 7 pasangan uji verifikasi
python3 evaluate.py --synthetic --sweep      # kalibrasi threshold (FAR/FRR)
python3 evaluate.py --identification         # uji 1-lawan-N (rank-1 accuracy)
```

Hasil aktual: **verifikasi 100% (FAR 0, FRR 0)** ·
**identifikasi rank-1 6/6 = 100%, jari asing 0/1 lolos**.

---

## 🧩 Struktur Project

```
├── app.py               # 🌐 Dashboard web (Streamlit) — opsi demo utama
├── main.py              # GUI OpenCV — opsi demo kedua
├── preprocessing.py     # LS 1-4: grayscale, CLAHE, GaussianBlur, unsharp, noise
├── segmentation.py      # LS 6-7: Otsu/adaptive threshold, opening/closing
├── matcher.py           # ORB + BFMatcher + ratio test + skor + keputusan
├── identification.py    # Mode 1-lawan-N: scan database, ranking, pemilik
├── evaluate.py          # Uji batch + sweep kalibrasi + uji identifikasi
├── generate_samples.py  # Generator sidik jari sintetis + database pengguna
├── samples/             # 7 citra sampel (rotasi, noise, blur, asing, dll.)
├── database/            # Pengguna terdaftar: Andi.png, Budi.png, Citra.png
└── DOKUMENTASI.md       # 📖 Penjelasan lengkap: teori, kode, hasil uji, demo
```

---

## 📚 Materi Labsheet yang Dipakai

| Labsheet | Materi | Peran |
|---|---|---|
| LS 1–2 | I/O, grayscale, resize | Fondasi input |
| **LS 3** ⭐ | **Ekualisasi Histogram & CLAHE** | Kontras ridge per tile + clipLimit (materi inti) |
| LS 4 | GaussianBlur, unsharp masking | Redam noise sebelum matching |
| LS 6 | Opening/Closing | Bersihkan citra biner |
| LS 7 | Otsu / Adaptive threshold | Binarisasi & panel tahapan |
| Baru | ORB + BFMatcher + ratio test | Jantung pencocokan |

Penjelasan lengkap (dengan contoh angka & analogi): **[DOKUMENTASI.md](DOKUMENTASI.md)**.

---

## 🖼️ Dataset asli (opsional, untuk demo final)

[**SOCOFing**](https://www.kaggle.com/datasets/ruizgara/socofing) dari Kaggle
(±1 GB). Ekstrak ke `dataset/` lalu:

```bash
python3 evaluate.py --dir dataset/Real --pairs 100 --sweep
python3 main.py "dataset/Real/1__M_Left_index_finger_CR.BMP" \
                "dataset/Altered/Altered-Hard/1__M_Left_index_finger_Zcut.BMP"
```

---

## 🧰 Troubleshooting cepat

| Gejala | Solusi |
|---|---|
| `The function is not implemented ... namedWindowImpl` (saat pakai `main.py`) | `pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python` (dashboard `app.py` tidak terdampak) |
| `Citra tidak ditemukan` | Jalankan `python3 generate_samples.py` dulu |
| Browser tidak terbuka | Buka manual `http://localhost:8501` |

---

*Politeknik Artificial Intelligence Budi Mulia Dua — Praktikum Computer Vision (WR25224/WR25334).*
