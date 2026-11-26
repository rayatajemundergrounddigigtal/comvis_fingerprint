# ============================================================
# app.py — DASHBOARD WEB Fingerprint Matcher (Streamlit)
# UAS Praktikum Computer Vision — Semester 3
#
# Desain modern dark-gradient, fitur lengkap:
#   Tab 1 — VERIFIKASI (1-lawan-1)  : upload 2 gambar → COCOK/TIDAK
#   Tab 2 — IDENTIFIKASI (1-lawan-N): upload 1 gambar → cari pemilik
#   Tab 3 — TENTANG                 : materi + cara kerja sistem
#
# Menjalankan:
#   cd ~/fingerprint_matcher
#   streamlit run app.py
# Browser terbuka otomatis di http://localhost:8501
#
# Semua pemrosesan memakai fungsi pipeline yang sama dengan main.py
# (tidak ada logika yang diduplikasi): enhance, segmentasi_lengkap,
# cocokkan, identifikasi.
# ============================================================

import os

import cv2
import numpy as np
import streamlit as st

from preprocessing import load_grayscale, enhance
from segmentation import segmentasi_lengkap
from matcher import cocokkan, keputusan
from identification import identifikasi, daftar_file_db

# ---------------- Konfigurasi halaman ----------------
st.set_page_config(page_title="Fingerprint Matcher UAS",
                   page_icon="🫆", layout="wide")

# ---------------- CSS desain modern ----------------
st.markdown("""
<style>
/* ====== Tema global: gelap + aksen gradien ====== */
.stApp {
    background: radial-gradient(1200px 600px at 80% -10%,
                rgba(56, 189, 248, .13), transparent 60%),
                radial-gradient(900px 500px at 10% 110%,
                rgba(168, 85, 247, .12), transparent 60%),
                #0e1117;
    color: #e6e9ef;
}
#MainMenu, footer, header {visibility: hidden;}

/* Judul dengan gradien */
.hero-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(90deg, #38bdf8, #a78bfa 55%, #f472b6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin-bottom: 0.1rem;
}
.hero-sub {color: #9aa4b2; font-size: 1.02rem; margin-bottom: 0.4rem;}

/* Kartu kaca (glassmorphism) */
.glass {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px; padding: 18px 20px;
    backdrop-filter: blur(10px);
}
.glass + .glass {margin-top: 14px;}

/* Chip status keputusan */
.chip-ok, .chip-no {
    display:inline-block; padding: 10px 22px; border-radius: 999px;
    font-weight: 800; font-size: 1.05rem; letter-spacing: .4px;
    border: 1px solid transparent;
}
.chip-ok {background: rgba(16,185,129,.14); color: #34d399; border-color: rgba(16,185,129,.4);}
.chip-no {background: rgba(239,68,68,.13);  color: #f87171; border-color: rgba(239,68,68,.4);}

/* Metrik besar */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 12px 16px;
}
div[data-testid="stMetricLabel"] {color:#9aa4b2;}
div[data-testid="stMetricValue"] {color:#f8fafc;}

/* Slider & tombol aksen */
.stSlider > div > div > div > div {background: linear-gradient(90deg,#38bdf8,#a78bfa);}
.stButton > button {
    border-radius: 12px; font-weight: 700;
    background: linear-gradient(90deg,#38bdf8,#a78bfa); color: #0b1220;
    border: none; padding: 10px 24px;
}
.stButton > button:hover {filter: brightness(1.08);}

/* Panel gambar */
.stImage img {border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);}

/* Tab */
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 8px 18px;
    background: rgba(255,255,255,0.03);
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,.14); color: #7dd3fc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar: kontrol parameter ----------------
with st.sidebar:
    st.markdown("## ⚙️ Kontrol Pipeline")
    st.caption("Geser — hasil ikut berubah. Sama seperti trackbar di GUI OpenCV.")

    threshold = st.slider("🚦 Ambang keputusan (%)", 1, 40, 7,
                          help="Skor ≥ ambang → COCOK. Rendah = jari asing bisa lolos "
                               "(FAR naik). Tinggi = pemilik asli bisa ditolak (FRR naik).")
    clip = st.slider("✨ Kekuatan CLAHE (clipLimit)", 1.0, 8.0, 3.0, 0.5,
                     help="LS 3 — kontras lokal per tile. Kecil = datar, besar = noise ikut kuat.")
    tile_grid = st.slider("🧩 Grid CLAHE (jumlah tile)", 4, 16, 8, 2,
                          help="LS 3 — jumlah 'kamar' pemecahan. Besar = adaptasi lebih lokal.")
    unsharp = st.slider("🔪 Unsharp masking (k)", 0.0, 1.5, 0.0, 0.1,
                        help="LS 4 — penajaman opsional. Default mati: CLAHE sudah cukup.")

    st.divider()
    st.markdown("### 👁️ Tahapan yang ditampilkan")
    tampil = st.pills("Tahapan",
                      ["Asli", "CLAHE", "Filter", "Biner", "Morfologi"],
                      selection_mode="multi",
                      default=["Asli", "CLAHE", "Filter", "Biner", "Morfologi"],
                      label_visibility="collapsed")

    st.divider()
    st.markdown("### ℹ️ Trade-off ambang")
    st.markdown(
        f"""
        | Ambang | Dampak |
        |---|---|
        | `{threshold}%` | {"🔓 longgar — FAR naik" if threshold < 4 else "🔒 rapat — FRR naik" if threshold > 12 else "⚖️ zona optimal (4–12%)"} |
        """)
    st.caption("Default 7% = hasil kalibrasi FAR/FRR "
               "(`python3 evaluate.py --synthetic --sweep`).")

# ---------------- Util ----------------


def baca_upload(uploaded_file, ukuran=(400, 400)):
    """File dari browser → citra grayscale 400x400."""
    data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, ukuran)


def kotak(img, judul):
    """Panel satu tahapan berlabel (versi web dari main.py)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (220, 220))
    label = np.full((26, 220, 3), 24, np.uint8)
    cv2.putText(label, judul, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 220, 255), 1)
    return np.vstack([label, img])


def urutan_tahapan(tahap, seg, prefix=""):
    """Daftar (label, gambar) sesuai pill yang dipilih user."""
    semua = [
        ("Asli", kotak(tahap["asli"], f"{prefix}1. Asli (LS1-2)")),
        ("CLAHE", kotak(tahap["clahe"], f"{prefix}2. CLAHE (LS3)")),
        ("Filter", kotak(tahap["enhanced"], f"{prefix}3. Filter (LS4)")),
        ("Biner", kotak(seg["biner"], f"{prefix}4. Biner (LS7)")),
        ("Morfologi", kotak(seg["morfologi"], f"{prefix}5. Morfologi (LS6)")),
    ]
    return [g for k, g in semua if k in tampil]


def tampilkan_baris_tahapan(tahap, seg, prefix="", kunci=""):
    """Baris panel tahapan (grid agar rapi & responsif)."""
    imgs = urutan_tahapan(tahap, seg, prefix)
    if not imgs:
        return
    kolom = st.columns(len(imgs), gap="small")
    for kol, gambar in zip(kolom, imgs):
        kol.image(gambar, use_container_width=True)


def chip(cocok, teks_ok="COCOK — AKSES DITERIMA", teks_no="TIDAK COCOK — AKSES DITOLAK"):
    st.markdown(f'<span class="chip-{"ok" if cocok else "no"}">'
                f'{"✅" if cocok else "❌"} {" " + teks_ok if cocok else " " + teks_no}'
                f'</span>', unsafe_allow_html=True)


def metrik_skor(skor, cocok_jumlah, kp):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Skor kemiripan", f"{skor:.1f}%")
    c2.metric("🚦 Ambang", f"{threshold:.0f}%")
    c3.metric("🔗 Titik cocok", cocok_jumlah)
    c4.metric("📍 Keypoint ORB", kp)


# ---------------- Hero ----------------
st.markdown('<div class="hero-title">🫆 Fingerprint Matcher</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub">UAS Praktikum Computer Vision — Semester 3 · '
            'Pipeline: Grayscale → <b>CLAHE</b> → Filter → Threshold → Morfologi → '
            '<b>ORB Matching</b> → Keputusan</div>', unsafe_allow_html=True)

# Peringatan database kosong (untuk tab identifikasi)
db_files = daftar_file_db("database/") if os.path.isdir("database/") else []

tab1, tab2, tab3 = st.tabs(
    ["🔐 Verifikasi (1-lawan-1)", "🔍 Identifikasi (1-lawan-N)", "📚 Tentang & Materi"])

# ================= TAB 1: VERIFIKASI =================
with tab1:
    st.markdown("*Pertanyaan yang dijawab: **apakah jari ini milik orang yang sama?***")

    col_q, col_r = st.columns(2)
    with col_q:
        with st.container(border=True):
            st.markdown("#### 📤 Gambar UJI (query)")
            up_q = st.file_uploader("Pilih gambar jari yang dites",
                                    type=["png", "jpg", "jpeg", "bmp", "tif"], key="q")
            if up_q:
                st.success(f"✓ {up_q.name}")
            else:
                st.caption("💡 Contoh: `samples/jari_A.png`")
    with col_r:
        with st.container(border=True):
            st.markdown("#### 📤 Gambar REFERENSI (terdaftar)")
            up_r = st.file_uploader("Pilih gambar pembanding",
                                    type=["png", "jpg", "jpeg", "bmp", "tif"], key="r")
            if up_r:
                st.success(f"✓ {up_r.name}")
            else:
                st.caption("💡 Contoh: `samples/jari_A_rotasi.png`")

    if up_q and up_r:
        q_img = baca_upload(up_q)
        r_img = baca_upload(up_r)
        if q_img is None or r_img is None:
            st.error("File tidak bisa dibaca sebagai gambar. Gunakan PNG/JPG/BMP.")
        else:
            with st.spinner("Memproses pipeline... CLAHE → filter → ORB → matching"):
                q_tahap = enhance(q_img, clip_limit=clip,
                                  tile_grid=(tile_grid, tile_grid), unsharp_k=unsharp)
                r_tahap = enhance(r_img, clip_limit=clip,
                                  tile_grid=(tile_grid, tile_grid), unsharp_k=unsharp)
                q_seg = segmentasi_lengkap(q_tahap["enhanced"])
                r_seg = segmentasi_lengkap(r_tahap["enhanced"])
                hasil = cocokkan(q_tahap["enhanced"], r_tahap["enhanced"])
                cocok = keputusan(hasil["skor"], threshold)

            st.markdown('<div class="glass">', unsafe_allow_html=True)
            chip(cocok)
            metrik_skor(hasil["skor"], len(hasil["good_matches"]),
                        f"{hasil['jumlah_kp'][0]} vs {hasil['jumlah_kp'][1]}")
            if hasil["gambar_match"] is not None:
                st.image(cv2.cvtColor(hasil["gambar_match"], cv2.COLOR_BGR2RGB),
                         caption="🔗 Garis hijau = titik yang cocok (ratio test 0.75)",
                         use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### 🧪 Tahapan pemrosesan")
            st.markdown("**Query**")
            tampilkan_baris_tahapan(q_tahap, q_seg, prefix="Q")
            st.markdown("**Referensi**")
            tampilkan_baris_tahapan(r_tahap, r_seg, prefix="R")
    else:
        st.info("ℹ️ Upload **kedua** gambar di atas — hasil muncul otomatis.")

# ================= TAB 2: IDENTIFIKASI =================
with tab2:
    st.markdown("*Pertanyaan yang dijawab: **jari ini milik siapa?*** "
                "Sistem membandingkan query ke semua pengguna di folder `database/`.")

    if not db_files:
        st.warning("Database masih kosong. Jalankan `python3 generate_samples.py` "
                   "di terminal sekali untuk membuat pengguna: Andi, Budi, Citra.")
    else:
        st.caption(f"📁 {len(db_files)} pengguna terdaftar: " +
                   ", ".join(os.path.splitext(os.path.basename(f))[0]
                             for f in db_files))

    col_up, col_db = st.columns([2, 1])
    with col_up:
        with st.container(border=True):
            st.markdown("#### 📤 Upload 1 gambar jari")
            up_id = st.file_uploader("Pilih gambar yang akan dicari pemiliknya",
                                     type=["png", "jpg", "jpeg", "bmp", "tif"], key="id")
            if up_id:
                st.success(f"✓ {up_id.name}")
    with col_db:
        folder_db = st.text_input("📁 Folder database", "database/")
        jalankan = st.button("🔍 CARI PEMILIKNYA", type="primary", use_container_width=True)

    if up_id and jalankan:
        if not os.path.isdir(folder_db):
            st.error(f"Folder `{folder_db}` tidak ada.")
        else:
            q_img = baca_upload(up_id)
            if q_img is None:
                st.error("File tidak bisa dibaca sebagai gambar.")
            else:
                with st.spinner("Memindai database..."):
                    tmp = ".query_tmp.png"
                    cv2.imwrite(tmp, q_img)
                    idn = identifikasi(tmp, folder_db, threshold,
                                       clip_limit=clip, unsharp_k=unsharp)
                    os.remove(tmp)

                terbaik = idn["terbaik"]
                cocok = idn["dikenal"]

                st.markdown('<div class="glass">', unsafe_allow_html=True)
                chip(cocok,
                     teks_ok=f"PEMILIK: {terbaik['nama'].upper()} — AKSES DITERIMA",
                     teks_no=f"TIDAK DIKENAL (terdekat: {terbaik['nama']}) — AKSES DITOLAK")
                c1, c2 = st.columns(2)
                c1.metric("🎯 Skor terbaik", f"{terbaik['skor']:.1f}%")
                c2.metric("👥 Pengguna diperiksa", len(idn["ranking"]))

                st.markdown("##### 📊 Ranking kandidat")
                for i, r in enumerate(idn["ranking"]):
                    medali = "🥇" if i == 0 else f"{i+1}."
                    teks = (f"{medali} **{r['nama']}** — {r['skor']:.1f}%"
                            + ("  ◄ PEMILIK" if i == 0 and cocok else ""))
                    st.progress(min(r["skor"], 100) / 100, text=teks)
                st.markdown('</div>', unsafe_allow_html=True)

                if terbaik["hasil"]["gambar_match"] is not None:
                    st.image(cv2.cvtColor(terbaik["hasil"]["gambar_match"],
                                          cv2.COLOR_BGR2RGB),
                             caption=f"🔗 Query vs kandidat terbaik ({terbaik['nama']})",
                             use_container_width=True)

                st.markdown("#### 🧪 Tahapan pemrosesan")
                q_tahap = idn["query_tahap"]
                r_tahap = terbaik["tahap"]
                q_seg = segmentasi_lengkap(q_tahap["enhanced"])
                r_seg = segmentasi_lengkap(r_tahap["enhanced"])
                st.markdown("**Query (jari yang dicari)**")
                tampilkan_baris_tahapan(q_tahap, q_seg, prefix="Q")
                st.markdown(f"**{terbaik['nama']} (kandidat terbaik)**")
                tampilkan_baris_tahapan(r_tahap, r_seg, prefix=terbaik["nama"] + " — ")

# ================= TAB 3: TENTANG =================
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🔁 Verifikasi vs Identifikasi")
        st.markdown("""
| Mode | Pertanyaan | Butuh |
|---|---|---|
| **Verifikasi** (1-lawan-1) | "Apakah jari ini milik **X**?" | 2 gambar |
| **Identifikasi** (1-lawan-N) | "Jari ini milik **siapa**?" | 1 gambar + database |
""")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 📚 Peta Materi Labsheet")
        st.markdown("""
| LS | Materi | Peran di sini |
|---|---|---|
| 1–2 | I/O, grayscale, resize | Fondasi input |
| **3** | **Histogram & CLAHE** ⭐ | **Kontras ridge per tile + clipLimit** |
| 4 | GaussianBlur, unsharp | Redam noise sebelum matching |
| 6 | Opening/Closing | Bersihkan citra biner (panel) |
| 7 | Otsu/Adaptive | Binarisasi (panel) |
| Baru | ORB + BFMatcher + ratio test | Jantung pencocokan |
""")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Cara kerja (versi singkat)")
        st.markdown("""
1. **Baca & bersihkan** — gambar jadi grayscale 400×400.
2. **CLAHE (LS 3)** ⭐ — kontras diperbaiki *per tile 8×8* dengan pembatas
   `clipLimit`, supaya ridge yang pudar tetap terbaca tanpa memperkuat noise.
3. **Filter (LS 4)** — Gaussian blur meredam noise halus.
4. **ORB** — tandai ±1000 titik unik (percabangan, ujung garis) + descriptor
   biner 256 bit yang tahan rotasi.
5. **BFMatcher + ratio test 0.75** — hitung pasangan titik yang yakin cocok.
6. **Skor** = titik cocok ÷ keypoint × 100%. ≥ ambang → **AKSES DITERIMA**.
""")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🗣️ Analogi 30 detik")
        st.markdown("""
> *"Program ini seperti detektif sidik: foto jari dulu 'dipercantik' per
> wilayah dengan CLAHE (tiap kamar dapat pencahayaan pas — dibatasi supaya
> debunya tidak ikut terlihat). Lalu ia menandai landmark unik di dua peta
> dan menghitung berapa yang sama. Banyak yang sama → jari yang sama."*
""")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("🖥️ GUI OpenCV alternatif: `python3 main.py` · "
           "Uji batch: `python3 evaluate.py --synthetic --sweep` · "
           "Dokumentasi lengkap: `DOKUMENTASI.md`")