# 📖 DOKUMENTASI LENGKAP — FINGERPRINT MATCHER
### UAS Praktikum Computer Vision — Semester 3
### Politeknik Artificial Intelligence Budi Mulia Dua

> **Cara membaca dokumen ini**: Bagian 1–3 ditulis dengan bahasa yang bisa
> dipahami siapa saja (bahkan orang yang belum pernah belajar pemrograman).
> Bagian dalam kotak [🔧 Teknis] berisi detail untuk laporan/dosen — boleh
> dibaca belakangan. **Materi inti yang paling ditonjolkan adalah
> EKUALISASI HISTOGRAM & CLAHE (Bagian 3)**, karena itulah "mesin utama"
> yang membuat garis sidik jari terbaca.

---

## DAFTAR ISI
1. [Project Ini Itu Apa? (Penjelasan 1 Menit)](#1-project-ini-itu-apa-penjelasan-1-menit)
2. [Cara Kerja Sistem, Langkah demi Langkah](#2-cara-kerja-sistem-langkah-demi-langkah)
3. [⭐ MATERI INTI: Ekualisasi Histogram & CLAHE](#3--materi-inti-ekualisasi-histogram--clahe)
4. [Materi Pendukung (LS 1, 2, 4, 6, 7 + ORB)](#4-materi-pendukung-ls-1-2-4-6-7--orb)
5. [Bedah Kode per File](#5-bedah-kode-per-file)
6. [Hasil Pengujian & Kalibrasi Threshold](#6-hasil-pengujian--kalibrasi-threshold)
7. [Skrip Demo Presentasi](#7-skrip-demo-presentasi)
8. [Antisipasi Pertanyaan Dosen](#8-antisipasi-pertanyaan-dosen)
9. [Panduan Dataset SOCOFing](#9-panduan-dataset-socofing)
10. [Troubleshooting](#10-troubleshooting)
11. [Referensi](#11-referensi)

---

## 1. PROJECT INI ITU APA? (Penjelasan 1 Menit)

Bayangkan sistem absensi kantor yang membaca sidik jari. Saat jari
ditempel, sistem harus menjawab dua pertanyaan:

| Mode | Pertanyaan | Contoh |
|---|---|---|
| **Verifikasi** (1-lawan-1) | "Apakah jari ini jari **Andi**?" | Kunci pintu rumah: cocok/tidak |
| **Identifikasi** (1-lawan-N) | "Jari ini milik **siapa**?" | Absensi: cari pemiliknya di daftar karyawan |

Project ini membangun keduanya dengan OpenCV:

- Kamu memberi **1 gambar jari** (mode identifikasi) atau **2 gambar jari**
  (mode verifikasi).
- Sistem membersihkan dan mempertajam gambarnya.
- Sistem menandai titik-titik unik di kedua gambar, lalu menghitung
  **berapa persen titik yang cocok**.
- Skornya tinggi → **AKSES DITERIMA** (dan menampilkan nama pemilik).
  Skornya rendah → **AKSES DITOLAK**.

> 🗣️ **Kalimat sakti untuk orang awam**: *"Program ini seperti membandingkan
> dua foto wajah — tapi yang dibandingkan polanya garis-garis jari. Supaya
> garisnya terlihat jelas, fotonya dulu 'dipercantik' dengan CLAHE. Lalu
> program menghitung: dari ribuan titik unik, berapa yang sama di kedua
> foto? Kalau banyak yang sama, berarti jari yang sama."*

---

## 2. CARA KERJA SISTEM, LANGKAH DEMI LANGKAH

Ini yang **terjadi secara berurutan** setiap kali sebuah jari diproses.
Analogi besarnya: sistem bekerja seperti **polisi sidik (forensik)** yang
memeriksa sidik jari di kertas — diperjelas, dicari ciri-cirinya, lalu
dicocokkan dengan arsip.

```
FOTO JARI (dari file)
   │
   ▼
[LANGKAH 1] Baca gambar, jadikan hitam-putih, samakan ukuran 400×400
   │         (warna dihapus — pola jari cuma butuh gelap-terang)
   ▼
[LANGKAH 2] PERCANTIK GAMBAR (bagian paling penting!)
   │         • CLAHE        → garis pudar dibuat tegas  (LS 3) ⭐
   │         • Gaussian Blur → bintik noise diratakan    (LS 4)
   ▼
   ├──► JALUR A "untuk dipamerkan": gambar diubah jadi hitam-putih
   │    murni (threshold LS 7) lalu dibersihkan (morfologi LS 6).
   │    Hasilnya ditampilkan di panel demo sebagai bukti proses.
   │
   ▼
[LANGKAH 3] CARI TITIK UNIK (ORB)
   │         Program menandai ±1000 titik unik: percabangan garis,
   │         ujung garis, persimpangan — seperti detektif menandai
   │         ciri khas sidik jari.
   ▼
[LANGKAH 4] COCOKKAN (BFMatcher)
   │         Setiap titik di gambar A dicari kembarannya di gambar B.
   │         Yang cocoknya yakin digambar garis hijau penghubung.
   ▼
[LANGKAH 5] HITUNG SKOR & PUTUSKAN
            skor % = (jumlah titik cocok) ÷ (jumlah titik) × 100
            skor ≥ ambang (7%) → COCOK / AKSES DITERIMA ✅
            skor < ambang      → TIDAK COCOK / DITOLAK ❌
```

### 2.1 Ilustrasi nyata (angka dari uji kami)

Mode identifikasi — query `jari_A_rotasi.png` (jari Andi yang diputar 15°):

```
                    SKOR COCOK
  database/Andi.png    21.3%   ◄── TERTINGGI → "PEMILIK: ANDI ✅"
  database/Budi.png     1.5%
  database/Citra.png    0.8%
  ─────────────────────────────
  ambang keputusan: 7%  ──┬── Andi (21.3%) di atas ambang → DIKENAL
                          └── Budi & Citra jauh di bawah → bukan dia
```

### 2.2 Berapa lama? (performa)

| Proses | Waktu kira-kira |
|---|---|
| Baca + perbaiki 1 gambar | ±50 md (milidetik) |
| Cari 1000 titik ORB | ±30 md |
| Cocokkan 2 gambar | ±10 md |
| Identifikasi: scan 3 pengguna | < 0,5 detik |
| Identifikasi: scan 100 pengguna | ±10 detik (masih real-time untuk demo) |

### 2.3 Dua jalur? Kenapa hasil hitam-putih tidak dipakai untuk mencocokkan?

Pertanyaan penting yang pasti muncul (dan pertanyaan favorit dosen).

> 🗣️ **Jawaban mudah**: *"Gambar hitam-putih hasil threshold itu seperti
> siluet — bentuknya jelas, tapi detailnya hilang semua. Padahal yang
> membedakan dua jari justru detail kecil. Jadi siluet hitam-putih itu
> kami pakai hanya untuk 'foto proses' di panel demo, sedangkan
> pencocokan memakai versi abu-abu yang sudah dipercantik CLAHE —
> di situlah detailnya masih hidup."*

---

## 3. ⭐ MATERI INTI: EKUALISASI HISTOGRAM & CLAHE

*(Ini materi LS 3. Dibahas paling dalam karena menjadi jantung project.)*

### 3.1 Mulai dari masalah: foto jari yang "tidak enak dibaca"

Coba perhatikan foto sidik jari hasil scan ponsel. Hampir selalu:
- **tengah jari lebih gelap** (ditekan lebih kuat ke kaca),
- **pinggir jari lebih pudar** (ditekan ringan),
- kadang ada **kilau cahaya** di satu sisi.

Padahal seluruh informasi jari ada di **garis-garis** itu (namanya
*ridge* = punggung garis, *valley* = celah antar garis). Kalau separuh
foto pudar dan separuh gelap, program kesulitan "membaca" garis.

**Solusinya: perbaiki kontras.** Dan di sinilah dua istilah penting:

> **Kontras** = beda tingkat terang antara bagian terterang dan
> tertgelap gambar. Kontras tinggi = hitam benar-benar hitam, putih
> benar-benar putih, garis terlihat tegas.
>
> **Histogram** = rekapitulasi "hitungan suara" kecerahan: berapa banyak
> piksel yang benar-benar gelap, berapa yang abu, berapa yang terang.
> Persis seperti rekap nilai ulangan kelas: berapa siswa dapat 60,
> berapa dapat 70, dst.

### 3.2 Ekualisasi Histogram (HE) — cara kerjanya, langkah demi langkah

*"Ekualisasi"* = **menyamakan sebarannya**. Idenya: kalau semua nilai
kecerahan menumpuk di rentang sempit (misal hanya 40–60), kita
**meregangkannya** supaya memenuhi rentang penuh 0–255. Hasilnya: gambar
yang tadinya "abu-abu mendung" jadi kontras.

**Contoh nyata dengan angka** (supaya tidak abstrak):

Ambil gambar mini 2×2 (4 piksel) dengan nilai: `40, 40, 50, 200`

```
LANGKAH 1 — Hitung histogram (rekap frekuensi):
   nilai 40 : muncul 2×  → peluang 2/4 = 0.50
   nilai 50 : muncul 1×  → peluang 1/4 = 0.25
   nilai 200: muncul 1×  → peluang 1/4 = 0.25

LANGKAH 2 — Hitung CDF (peluang kumulatif, "total sementara"):
   40  → 0.50
   50  → 0.50 + 0.25 = 0.75
   200 → 0.75 + 0.25 = 1.00

LANGKAH 3 — Petakan ke rentang baru: nilai_baru = CDF × 255
   40  → 0.50 × 255 = 128
   50  → 0.75 × 255 = 191
   200 → 1.00 × 255 = 255
```

Hasil: `40, 40, 50, 200` → **`128, 128, 191, 255`**

Lihat apa yang terjadi: kelompok gelap (40–50) yang tadinya berdempetan
kini ter Regents dari 128–191, dan yang terang naik ke 255. **Sebaran
menjadi lebar → kontras naik.** Itulah ekualisasi histogram.

> 🗣️ **Analogi kelas**: bayangkan nilai ulangan seluruh kelas menumpuk di
> 40–50. Ekualisasi seperti "memetakan ulang" nilai secara adil berdasar
> peringkat kumulatif — siswa terlemah tetap di urutan bawah, tapi
> skornya diregangkan mengikuti skala penuh. Yang penting: **urutan tidak
> berubah**, hanya jarak antar nilai yang dibentangkan.

**Di OpenCV, satu baris:**
```python
hasil = cv2.equalizeHist(gray)     # HE global
```

### 3.3 Kenapa HE biasa tidak cukup? (di sinilah masalahnya)

HE memakai **SATU rumus pemetaan untuk SELURUH gambar** (global). Untuk
sidik jari, ini menimbulkan dua masalah:

1. **Bagian yang sudah terang jadi putih menyala (overexposed).**
   Pinggir jari yang kontrasnya sudah lumayan ikut "diregangkan" →
   detailnya hilang tertelan putih.
2. **Noise ikut diperkuat.** Bintik debu/gesekan kaca juga abu-abu;
   HE memperbesar perbedaannya → bintik jadi semakin jelas, seperti
   garis palsu.

> 🗣️ **Analogi**: HE itu seperti menaikkan volume **seluruh** lagu lewat
> satu tombol volume. Bagian yang pelan jadi terdengar — bagus — tapi
> bagian yang sudah keras jadi pecah, dan desis latar ikut keras.

### 3.4 CLAHE — HE yang "pintar": per ruangan, dengan pembatas

**CLAHE = Contrast Limited Adaptive Histogram Equalization.** Kata
panjangnya bisa dipecah jadi tiga ide sederhana:

| Kata | Artinya | Analogi |
|---|---|---|
| **Adaptive** | diperbaiki **per wilayah kecil**, bukan seluruh foto sekaligus | lampu tiap kamar diatur sendiri-sendiri |
| **Contrast Limited** | ada **pembatas penguatan** supaya noise tidak ikut kuat | mikrofon dengan pembatas volume otomatis |
| **Histogram Equalization** | meregangkan sebaran kecerahan | seperti contoh angka di 3.2 |

**Cara kerja CLAHE, langkah demi langkah:**

```
LANGKAH 1 — Bagi gambar menjadi kotak-kotak kecil (TILE)
   Default OpenCV: tileGridSize=(8,8) → gambar dibagi 8×8 = 64 "kamar".
   (Sidik jari 400×400 → tiap kamar 50×50 piksel.)

LANGKAH 2 — Tiap kamar punya histogram sendiri
   Kamar yang gelap (tengah jari) punya sebaran berbeda dengan kamar
   yang pudar (pinggir jari). Masing-masing dihitung sendiri.

LANGKAH 3 — POTONG histogram yang terlalu tinggi (clipLimit!) ⭐
   Kalau satu nilai kecerahan mendominasi sebuah kamar (histogram
   menjulang), itu tanda-tanda polos atau noise.
   → Puncak histogram DIPOTONG pada ketinggian clipLimit.
   → "Kelebihan penduduknya" dibagi rata ke semua tingkat kecerahan
     lain di kamar itu.
   Efeknya: kemiringan pemetaan jadi terbatas → penguatan kontras
   TIDAK meledak → noise tidak ikut diperkuat.

LANGKAH 4 — Ekualisasi tiap kamar (seperti 3.2, tapi per kamar)

LANGKAH 5 — Hapus jahitan antar kamar (interpolasi bilinear)
   Tanpa ini, foto tampak seperti mosaik bersambung.
   OpenCV mencampur hasil di perbatasan antar kamar secara halus.
```

> 🗣️ **Analogi lengkap CLAHE**: *"Bayangkan rumah 64 kamar yang pencahayaannya
> kacau — ada kamar remang, ada kamar silau. HE itu seperti memasang satu
> saklar dimmer untuk seluruh rumah: kamar remang terbantu, kamar silau
> makin menyiksa. CLAHE memasang dimmer di tiap kamar sesuai kondisi
> kamar itu, TAPI dengan aturan: dimmer dibatasi maksimalnya (clipLimit)
> supaya kamar yang kotor debunya tidak ikut terlihat mencolok. Terakhir,
> dinding antar kamar di-dempul halus supaya tidak terlihat batas kotaknya."*

**Di OpenCV, dua baris:**
```python
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
hasil = clahe.apply(gray)
```

### 3.5 Dua tombol aturannya: apa efeknya kalau digeser?

Ini yang kamu tunjukkan **live lewat trackbar** saat demo:

| Parameter | Digeser ke... | Efek terlihat |
|---|---|---|
| `clipLimit` (default 3.0; trackbar "ClipLimit x10" = 30) | **besar** (mis. 8) | kontras sangat kuat → ridge tegas, TAPI noise/bintik ikut mencolok |
| | **kecil** (mis. 1) | gambar datar, ridge kembali pudar |
| `tileGridSize` | **besar** (mis. 16×16, kamar lebih kecil) | adaptasi sangat lokal, detail halus keluar, tapi bisa muncul pola "mosaik" |
| | **kecil** (mis. 2×2, kamar besar) | makin mirip HE global — masalah lama kembali |

> 💡 **Uji coba mandiri 2 menit** (pas untuk praktikum): jalankan
> `python3 main.py`, geser trackbar ClipLimit dari 10 → 30 → 80 sambil
> memperhatikan kolom "2. CLAHE (LS3)" di panel. Kamu akan melihat persis
> tabel di atas. *Mengalami sendiri = hafal seumur hidup.*

### 3.6 Kenapa CLAHE sangat menentukan di project ini?

- Input matching adalah **ORB**, yang mencari titik unik dari perbedaan
  kecerahan. Kalau ridge pudar → titik tidak terdeteksi → skor rendah.
- Dengan CLAHE, ridge di area gelap maupun pudar sama-sama tegas →
  jumlah & kualitas keypoint naik → skor jari yang sama jadi jauh lebih
  tinggi dari jari orang lain → **pemisahan match/tidak-match jadi jelas**.
- Bukti di hasil uji kami: pasangan jari sama yang blur/noise tetap
  COCOK setelah enhancement; tanpa tahap ini, banyak yang tertolak.

### 3.7 Unsharp Masking: sahabat CLAHE (materi LS 4)

Setelah CLAHE membuat kontras merata, ada satu tahap opsional:
**penajaman (sharpening)**.

```
blur   = GaussianBlur(gambar)          # versi buram (hilang detail)
mask   = gambar − blur                 # "sisa" = detail halus saja
hasil  = gambar + k × mask             # detail ditumpuk kembali, diperkuat
```

> 🗣️ **Analogi**: seperti menyalin sketsa pensil lalu menelusuri ulang
> garisnya dengan spidol — garis besar sudah ada (gambar asli), kita
> hanya menambah ketegasan (detail).

Di project ini unsharp **default mati (k=0)** dan bisa dinyalakan lewat
trackbar, karena CLAHE sudah cukup — dan penajaman berlebihan justru
memperkuat noise. Ini contoh keputusan desain yang bisa kamu ceritakan
saat demo.

---

## 4. MATERI PENDUKUNG (LS 1, 2, 4, 6, 7 + ORB)

Tiap materi diberi format sama: **satu kalimat inti → cara kerja →
analogi**. (LS 3 sudah dibahas mendalam di Bagian 3.)

### LS 1 — Membaca & menyimpan citra
- **Inti**: gambar = kumpulan angka; OpenCV membacanya dengan `imread`.
- **Cara kerja**: file PNG/JPG diubah jadi matriks angka yang bisa dihitung.
- 🗣️ *"Bagi komputer, foto itu bukan gambar — itu tabel angka raksasa.
  Setiap titik punya angka kecerahan 0–255."*

### LS 2 — Grayscale & resize
- **Inti**: buang warna (jadi hitam-putih-abu), samakan ukuran.
- **Cara kerja**: tiap piksel warna (3 angka: merah-hijau-biru) diubah
  jadi 1 angka kecerahan. Ukuran disamakan 400×400.
- 🗣️ *"Warna tidak membantu mengenali pola jari — yang penting gelap-
  terang garisnya. Membuang warna membuat hitungan 3× lebih ringan."*
- [🔧 Teknis] Rumus konversi: `Y = 0.299·R + 0.587·G + 0.114·B`
  (hijau diberi bobot terbesar karena mata manusia paling sensitif ke hijau).

### LS 4 — Filter untuk menghaluskan noise
- **Inti**: meratakan bintik kotor supaya tidak terbaca sebagai garis palsu.
- **Cara kerja (GaussianBlur)**: tiap piksel diganti **rata-rata berbobot**
  tetangganya — tetangga dekat berpengaruh besar, jauh kecil.
- **Cara kerja (MedianBlur)**: tiap piksel diganti **nilai tengah**
  tetangganya — satu bintik ekstrem tidak bisa mengubah hasil.
- 🗣️ *"Noise itu seperti satu siswa yang teriak di kelas yang tenang.
  Gaussian = meratakan suara semua siswa. Median = ambil suara yang
  paling umum, jadi satu teriakan tidak berpengaruh."*
- [🔧 Teknis] Kernel harus ukuran ganjil (3×3, 5×5) supaya punya titik
  pusat yang jelas (Keselamatan Kerja LS 4).

### LS 7 — Thresholding: jadi hitam-putih murni
- **Inti**: mengubah foto jadi dua warna saja (hitam = garis, putih = celah).
- **Cara kerja (Otsu)**: program mencari **satu angka ambang** terbaik
  otomatis dari histogram — yang paling memisahkan kelompok gelap dan terang.
- **Cara kerja (Adaptive)**: ambangnya **beda-beda per wilayah kecil**
  — jauh lebih adil untuk jari, karena tekanan tengah ≠ pinggir.
- 🗣️ *"Satu standar untuk semua orang itu tidak adil. Adaptive itu seperti
  standar kelulusan yang disesuaikan kondisi tiap sekolah."*
- Hasil jalur ini dipakai untuk **panel visualisasi**, bukan matching (lihat 2.3).

### LS 6 — Morfologi: membersihkan citra biner
- **Inti**: memperbaiki gambar hitam-putih dengan "mengikis" dan
  "menambal".
- **Cara kerja**: *Opening* (ikis lalu tebal kembali) menghapus bintik
  kecil; *Closing* (tebal lalu ikis) menyambung garis yang putus.
  Urutan wajib: **Opening dulu, baru Closing** — kalau terbalik, bintik
  noise ikut membesar (Keselamatan Kerja LS 6).
- 🗣️ *"Opening itu mengayak pasir: butir kecil lolos dan hilang.
  Closing itu menambal aspal: celah kecil di garis tersambung lagi."*

### ORB + BFMatcher + Ratio Test (materi baru — jantung pencocokan)
- **Inti**: menandai titik unik, lalu menghitung berapa yang sama.
- **Cara kerja**:
  1. **ORB** menandai ±1000 *keypoint* (titik unik: percabangan, ujung
     garis) dan membuat **descriptor** — sidik jari numerik 256 bit dari
     sekitar titik itu. Descriptor punya orientasi, jadi tahan rotasi.
  2. **BFMatcher** mencocokkan tiap descriptor gambar A ke gambar B
     (jarak = jumlah bit berbeda, "Hamming distance").
  3. **Ratio test 0.75**: match diterima hanya kalau kandidat terbaik
     **jauh lebih baik** daripada kandidat kedua. Kalau mirip kualitasnya,
     match-nya ambigu → dibuang. Ini penyaring "match palsu".
- 🗣️ *"Seperti membandingkan dua peta kota: tandai landmark unik
  (bundaran, pertigaan) di keduanya, lalu hitung berapa landmark yang
  sama. 80% landmark cocok → hampir pasti kota yang sama."*
- **Skor akhir**: `good matches ÷ min(jumlah keypoint keduanya) × 100%`
- **Keputusan**: skor ≥ threshold (default 7%, hasil kalibrasi data) → COCOK.

> **Far/FRR** (dipakai di evaluasi): **FAR** = orang asing lolos (masalah
> keamanan), **FRR** = pemilik asli ditolak (masalah kenyamanan). Geser
> threshold → yang satu turun yang lain naik. Tabel kalibrasinya di Bagian 6.

---

## 5. BEDAH KODE PER FILE

| File | Isi singkat | Materi |
|---|---|---|
| `preprocessing.py` | `load_grayscale` (LS 1–2), `enhance` = **CLAHE → GaussianBlur → unsharp opsional** (LS 3–4), pembangkit noise | inti |
| `segmentation.py` | `binarisasi` (Otsu/adaptive, LS 7), `bersihkan_morfologi` (opening→closing, LS 6) | jalur visual |
| `matcher.py` | `ekstrak_fitur` (ORB), `cocokkan` (BFMatcher + ratio test + skor), `keputusan` | baru |
| `identification.py` | `identifikasi`: scan folder `database/`, ranking skor, tentukan pemilik | baru |
| `main.py` | GUI demo: panel 10 kotak tahapan, trackbar (Threshold/ClipLimit/Unsharp), tombol `o r i v s q` | demo |
| `app.py` | 🌐 **Dashboard web Streamlit**: upload di browser → hasil otomatis; 3 tab (Verifikasi/Identifikasi/Tentang), slider sidebar, desain dark glassmorphism. Panggil fungsi pipeline yang sama — logika tidak diduplikasi | demo |
| `evaluate.py` | uji batch verifikasi + `--sweep` kalibrasi FAR/FRR + `--identification` rank-1 | laporan |
| `generate_samples.py` | membuat 7 citra jari sintetis + 3 pengguna database (Andi, Budi, Citra) | data latihan |

**Fungsi enhancement (jantung, materi Bagian 3) secara utuh:**
```python
def enhance(gray, clip_limit=3.0, tile_grid=(8, 8), blur_ksize=5, unsharp_k=0.0):
    # 1) CLAHE: kontras lokal per tile 8x8, dibatasi clipLimit  (LS 3)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    hasil_clahe = clahe.apply(gray)

    # 2) Gaussian blur: redam noise halus, kernel ganjil       (LS 4)
    hasil_blur = cv2.GaussianBlur(hasil_clahe, (k, k), 0)

    # 3) Unsharp masking opsional: asli + k*(asli - blur)      (LS 4)
    ...
    return {"asli": gray, "clahe": hasil_clahe, "blur": hasil_blur,
            "enhanced": hasil_akhir}   # "enhanced" = input matching
```

---

## 6. HASIL PENGUJIAN & KALIBRASI THRESHOLD

### 6.1 Verifikasi (threshold 7%)

| Query | Referensi | Skor | Keputusan | Seharusnya |
|---|---|---|---|---|
| jari_A | jari_A (identik) | 100.0% | COCOK | SAMA ✅ |
| jari_A | jari_A_rotasi (15°) | 22.7% | COCOK | SAMA ✅ |
| jari_A | jari_A_noise | 17.1% | COCOK | SAMA ✅ |
| jari_A | jari_A_blur | 8.8% | COCOK | SAMA ✅ |
| jari_A | jari_B | 0.9% | TIDAK | BEDA ✅ |
| jari_A | jari_C | 1.9% | TIDAK | BEDA ✅ |
| jari_B | jari_C | 2.5% | TIDAK | BEDA ✅ |

**Akurasi 100% · FAR 0% · FRR 0%** (7 pasangan).

### 6.2 Identifikasi 1-lawan-N (threshold 7%)

| Query | Dikenali sebagai | Skor terbaik | Status |
|---|---|---|---|
| jari_A | Andi | 100.0% | OK |
| jari_A_rotasi | Andi | 21.3% | OK |
| jari_A_noise | Andi | 16.5% | OK |
| jari_A_blur | Andi | 7.5% | OK |
| jari_B | Budi | 100.0% | OK |
| jari_C | Citra | 100.0% | OK |
| jari_asing (bukan pengguna) | — | 2.1% | Ditolak OK |

**Rank-1 accuracy 6/6 = 100% · jari asing lolos 0/1.**

### 6.3 Tabel kalibrasi threshold (`evaluate.py --synthetic --sweep`)

| Threshold | Akurasi | FAR | FRR |
|---|---|---|---|
| 2% | 85.7% | 33.3% | 0% |
| **4%** | **100%** | **0%** | **0%** |
| **6%** | **100%** | **0%** | **0%** |
| **7%** | **100%** | **0%** | **0%** |
| 8% | 100% | 0% | 0% |
| 10–16% | 85.7% | 0% | 25% |
| 18–22% | 71.4% | 0% | 50% |
| 24–30% | 57.1% | 0% | 75% |

**Interpretasi (bahasa sederhana)**: threshold 2% = pintu longgar →
jari orang lain bisa masuk (FAR 33%, bahaya!). Threshold 24%+ = pintu
terlalu rapat → jari asli yang sedikit blur ditolak (FRR 75%, menyebalkan!).
Zona emas **4–8%**; dipilih **7%** supaya jari_A_blur (7.5%) tetap
dikenali dan jari asing (2.1–2.5%) tertinggal jauh di bawahnya.

> Untuk SOCOFing, jalankan sweep ulang — angkanya akan berbeda, dan
> tabelnya dilampirkan di laporan sebagai bukti proses kalibrasi.

---

## 7. SKRIP DEMO PRESENTASI (±5 menit)

### 7.0 Opsi tampilan demo: Dashboard Web (Streamlit)

Selain GUI OpenCV, tersedia **dashboard web modern** (`app.py`) — upload
gambar langsung di browser, semua hasil muncul otomatis:

```bash
streamlit run app.py        # browser terbuka di http://localhost:8501
```

| Fitur dashboard | Isi |
|---|---|
| Tab **Verifikasi** | upload 2 gambar → banner ✅/❌ + metrik skor + garis match + panel tahapan |
| Tab **Identifikasi** | upload 1 gambar → klik "CARI PEMILIKNYA" → nama pemilik + bar ranking semua kandidat |
| Tab **Tentang & Materi** | peta materi labsheet + cara kerja + analogi 30 detik (panduan menjelaskan) |
| Sidebar | slider ambang/CLAHE/grid/unsharp — hasil update otomatis |

Desain dark-gradient glassmorphism; jendela bisa ditampilkan fullscreen di
proyektor (F11). Pemrosesan tetap lokal (localhost) — tidak ada data ke internet.
Momen trackbar (langkah 5–7 di bawah) di dashboard = geser slider di sidebar.

### 7.1 Alur demo (pakai dashboard ATAU GUI OpenCV)

1. **Buka GUI** (`python3 main.py`). Tunjuk panel:
   *"Kolom 2 adalah CLAHE dari LS 3 — materi inti kami. Kolom 3 filter
   LS 4, kolom 4 thresholding LS 7, kolom 5 morfologi LS 6. Semua tahap
   labsheet ada di layar."*
2. **Kasus 1 — jari sama, dirotasi**: default A vs A-rotasi → hijau COCOK.
   *"Diputar 15° tetap dikenali, karena descriptor ORB punya orientasi."*
3. **Kasus 2 — jari kotor**: tekan `r`, pilih `jari_A_noise.png` → COCOK.
   *"Noise salt-and-pepper & Gaussian diredam filter LS 4."*
4. **Kasus 3 — orang lain**: tekan `r`, pilih `jari_B.png` → merah DITOLAK.
5. **⭐ Momen CLAHE**: geser trackbar **ClipLimit x10** dari 30 → 10 → 80
   sambil menunjuk kolom "2. CLAHE": ridge melembut, tegas, lalu noise
   mencolok. *"Ini efek clipLimit — pembatas penguatan kontras. Persis
   teori: terlalu kecil kontras datar, terlalu besar noise ikut kuat."*
6. **Mode identifikasi**: tekan `i`, pilih `jari_asing.png` →
   "TIDAK DIKENAL" + ranking rendah. Lalu `i`, pilih `jari_A_rotasi.png`
   → **"PEMILIK: ANDI / AKSES DITERIMA"** + ranking top-5.
   *"Sekarang sistem menjawab 'jari ini milik siapa' — seperti absensi."*
7. **Trade-off threshold**: geser threshold ke 2 → jari B lolos!
   *"Threshold rendah = tidak aman (FAR naik). Tinggi = jari asli ditolak
   (FRR naik). Tabel kalibrasinya di laporan kami."* Kembalikan ke 7.
8. **Tutup** dengan hasil terminal `evaluate.py`: verifikasi 100%,
   identifikasi rank-1 100%.

Tips: tekan `s` untuk menyimpan panel sebagai screenshot cadangan bila
proyektor bermasalah.

---

## 8. ANTISIPASI PERTANYAAN DOSEN

**Soal CLAHE & histogram (materi inti — kuasai paling dalam):**

| Pertanyaan | Jawaban |
|---|---|
| Apa itu histogram citra? | Rekapitulasi frekuensi nilai kecerahan 0–255 — "rapor" distribusi terang-gelap gambar. |
| Apa itu ekualisasi histogram? | Meregangkan sebaran kecerahan yang menumpuk ke rentang penuh, via CDF (contoh angka: 40,40,50,200 → 128,128,191,255). |
| Bedanya HE dan CLAHE? | HE global (1 rumus untuk semua piksel); CLAHE per-tile (lokal) + clipLimit membatasi penguatan noise. |
| Apa fungsi clipLimit? | Memotong puncak histogram tiap tile; kelebihan didistribusikan ulang → kemiringan pemetaan terbatas → noise tidak diperkuat. |
| Apa efek tileGridSize? | Tile kecil-kecil = adaptasi lebih lokal tapi risiko artefak mosaik; tile besar = mendekati HE global. |
| Kenapa CLAHE, bukan HE biasa? | Tekanan jari tidak merata: HE membuat area terang overexposed dan noise ikut kuat; CLAHE memperbaiki tiap wilayah dengan pembatas. |
| Kenapa unsharp masking opsional? | CLAHE sudah cukup meningkatkan kontras; penajaman berlebihan memperkuat noise — default mati (k=0), trackbar untuk eksperimen. |

**Soal materi lain:**

| Pertanyaan | Jawaban |
|---|---|
| Kenapa tidak bandingkan piksel langsung? | Jari bisa bergeser/berotasi → piksel tidak sejajar. Keypoint matching tahan geser & rotasi. |
| Kenapa biner tidak dipakai matching? | Terlalu kasar — sudut palsu seragam. Biner untuk visualisasi & segmentasi; matching dari grayscale enhanced. |
| Otsu itu apa? | Ambang otomatis terbaik dari histogram (minim varians intra-kelas). Kenapa adaptive? Tekanan jari tak merata → ambang per wilayah. |
| Apa itu minutiae? | Ridge ending & bifurcation — ciri khas sidik jari; keypoint ORB menangkap struktur serupa. |
| Ratio test itu apa? | Match diterima hanya jika kandidat terbaik < 0.75× kandidat kedua; ambigu → buang. |
| Kenapa ORB bukan SIFT? | Lebih cepat (biner + Hamming), gratis, cukup akurat untuk 400×400. |
| Verification vs identification? | Verifikasi: "apakah ini jari X?" (1-lawan-1). Identifikasi: "jari ini milik siapa?" (1-lawan-N). FAR identifikasi naik dengan ukuran database → threshold lebih ketat. |
| Kenapa opening dulu baru closing? | Closing duluan membesarkan bintik noise (Keselamatan Kerja LS 6). |
| FAR/FRR itu apa? | FAR = asing diterima (keamanan); FRR = pemilik ditolak (kenyamanan). Trade-off terhadap threshold. |
| Kenapa threshold 7%, bukan 75%? | Threshold dikalibrasi dari data (tabel sweep); 75% hanya ilustrasi konsep awal. |

---

## 9. PANDUAN DATASET SOCOFING

1. Download: https://www.kaggle.com/datasets/ruizgara/socofing
   (butuh akun Kaggle; ukuran ±1 GB).
2. Struktur penting:
   - `SOCOFing/Real/` — 6000 citra asli, nama: `1__M_Left_index_finger_CR.BMP`
     (identitas jari = bagian sebelum `__`).
   - `SOCOFing/Altered/Altered-Easy|Medium|Hard/` — versi rotasi, blur,
     noise, Zcut. Suffix nama menunjukkan jenis alterasi.
3. Ekstrak ke `fingerprint_matcher/dataset/`.
4. Kalibrasi & uji:
   ```bash
   python3 evaluate.py --dir dataset/Real --pairs 100 --sweep
   python3 evaluate.py --dir dataset/Real --pairs 200 --threshold <hasil sweep>
   ```
5. Demo pasangan altered:
   ```bash
   python3 main.py "dataset/Real/1__M_Left_index_finger_CR.BMP" \
                   "dataset/Altered/Altered-Hard/1__M_Left_index_finger_Zcut.BMP"
   ```
6. **Database identifikasi dari SOCOFing**: salin ±20 file `Real` ke
   `database/` (satu file per orang supaya nama pemilik jelas), lalu:
   ```bash
   python3 main.py --db database/ "dataset/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP"
   ```
7. Lampirkan tabel sweep SOCOFing di laporan sebagai bukti kalibrasi
   pada data asli.

---

## 10. TROUBLESHOOTING

| Gejala | Penyebab | Solusi |
|---|---|---|
| `The function is not implemented ... namedWindowImpl` | `opencv-python-headless` terinstall bersamaan/menimpa versi GUI | `pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python` (sudah ditangani; `main.py` memberi pesan ramah via `cek_gui()`) |
| `Citra tidak ditemukan` | path salah / belum generate sampel | `python3 generate_samples.py`, atau jalankan dari folder project |
| Jendela tidak muncul di SSH | tidak ada display | jalankan dari sesi desktop, atau SSH dengan X forwarding |
| Skor semua pasangan rendah | citra terlalu polos/kecil | pastikan resize 400×400 & CLAHE aktif; cek `jumlah_kp` di banner |
| `tkinter` tidak ada (dialog file gagal) | python tanpa tk | `sudo apt install python3-tk`, atau pakai argumen CLI `python3 main.py file1 file2` |
| GUI lambat | recompute tiap perubahan trackbar | normal; panel hanya dihitung ulang saat parameter berubah |

---

## 11. REFERENSI

1. R. C. Gonzalez & R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018. *(bab histogram processing & equalization)*
2. K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", *Graphics Gems IV*, 1994. *(paper asal CLAHE)*
3. S. M. Pizer et al., "Adaptive histogram equalization and its variations", *Computer Vision, Graphics, and Image Processing*, 1987. *(AHE → CLAHE)*
4. R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., Springer, 2022.
5. J. Howse & J. Minichino, *Learning OpenCV 5 Computer Vision with Python*, 4th ed., Packt, 2024.
6. Labsheet Praktikum Computer Vision LS 1–7, Politeknik AI Budi Mulia Dua (WR25224/WR25334).
7. OpenCV documentation: *CLAHE*, *Histogram Equalization*, *Feature Matching*, *ORB*, *Morphological Transformations*, *Image Thresholding*.
8. E. Rublee et al., "ORB: An efficient alternative to SIFT or SURF", ICCV 2011.
9. D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004 (ratio test).
10. SOCOFing dataset: https://www.kaggle.com/datasets/ruizgara/socofing

---

*Dokumen ini dibuat bersamaan dengan kode project. Setiap kali kode diubah,
perbarui bagian hasil pengujian (Bagian 6) dengan menjalankan ulang
`python3 evaluate.py --synthetic --sweep`.*