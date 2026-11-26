# ============================================================
# main.py — GUI DEMO Fingerprint Matcher (UAS Semester 3)
#
# Cara pakai:
#   python3 main.py                                       -> verifikasi: sampel default
#   python3 main.py gambar1.png gambar2.png               -> verifikasi 1-lawan-1
#   python3 main.py --db database/ gambar1.png            -> identifikasi 1-lawan-N
#
# Kontrol saat demo:
#   Trackbar "Threshold %"  : ambang keputusan COCOK/TIDAK (geser live!)
#   Trackbar "ClipLimit x10": kekuatan CLAHE (LS 3) — geser, lihat efeknya
#   Trackbar "Unsharp x10"  : kekuatan penajaman (LS 4), 0 = mati
#   Tombol 'o' : pilih citra QUERY (jari yang dites)         [mode 1-lawan-1]
#   Tombol 'r' : pilih citra REFERENSI (jari terdaftar)      [mode 1-lawan-1]
#   Tombol 'i' : mode IDENTIFIKASI 1-lawan-N: pilih 1 jari,
#                sistem cari pemiliknya di folder database/
#   Tombol 'v' : kembali ke mode verifikasi (1-lawan-1)
#   Tombol 's' : simpan panel hasil ke file hasil_demo.png
#   Tombol 'q' : keluar
#
# Panel menampilkan SEMUA tahapan flow:
#   Asli -> CLAHE -> Filter -> Biner -> Morfologi -> Garis Match + Skor
# ============================================================

import argparse
import sys

import cv2
import numpy as np

from preprocessing import load_grayscale, enhance
from segmentation import segmentasi_lengkap
from matcher import cocokkan, keputusan
from identification import identifikasi

PANEL = 260          # ukuran tiap kotak tahapan di panel
JENDELA = "Fingerprint Matcher - UAS Computer Vision"


def pilih_file():
    """Dialog pilih file (tkinter). Return path atau None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Pilih citra sidik jari",
            filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp *.tif"), ("Semua", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


def kotak(img, judul, warna=(255, 255, 255)):
    """Buat satu panel tahapan: resize + label judul di atas."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (PANEL, PANEL))
    label = np.zeros((30, PANEL, 3), np.uint8)
    cv2.putText(label, judul, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna, 1)
    return np.vstack([label, img])


def susun_panel(q_tahap, r_tahap, q_seg, r_seg, hasil, threshold,
                judul_ref="1. Referensi Asli", status_teks=None, ranking=None):
    """Susun montage 2 baris x 5 kolom + gambar match + banner keputusan.

    Mode identification memakai judul_ref (nama kandidat), status_teks
    (nama pemilik / TIDAK DIKENAL), dan ranking (strip top-5).
    """
    baris1 = np.hstack([
        kotak(q_tahap["asli"], "1. Query Asli (LS1-2)"),
        kotak(q_tahap["clahe"], "2. CLAHE (LS3)"),
        kotak(q_tahap["enhanced"], "3. Filter (LS4)"),
        kotak(q_seg["biner"], "4. Biner (LS7)"),
        kotak(q_seg["morfologi"], "5. Morfologi (LS6)"),
    ])
    baris2 = np.hstack([
        kotak(r_tahap["asli"], judul_ref),
        kotak(r_tahap["clahe"], "2. CLAHE (LS3)"),
        kotak(r_tahap["enhanced"], "3. Filter (LS4)"),
        kotak(r_seg["biner"], "4. Biner (LS7)"),
        kotak(r_seg["morfologi"], "5. Morfologi (LS6)"),
    ])

    # Gambar garis match (tahap 5 flow), lebar disamakan dengan panel
    lebar = PANEL * 5
    gm = hasil["gambar_match"]
    if gm is None:
        gm = np.zeros((PANEL, lebar, 3), np.uint8)
        cv2.putText(gm, "Fitur tidak cukup untuk matching", (40, PANEL // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        tinggi = int(gm.shape[0] * lebar / gm.shape[1])
        gm = cv2.resize(gm, (lebar, tinggi))

    # Banner keputusan (tahap 6 flow)
    cocok = keputusan(hasil["skor"], threshold)
    banner = np.zeros((70, lebar, 3), np.uint8)
    warna = (0, 200, 0) if cocok else (0, 0, 230)
    if status_teks is None:
        status_teks = ("COCOK / AKSES DITERIMA" if cocok
                       else "TIDAK COCOK / AKSES DITOLAK")
    teks = f"Skor: {hasil['skor']:.1f}%  |  Threshold: {threshold:.0f}%  |  {status_teks}"
    cv2.putText(banner, teks, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, warna, 2)
    n_good = len(hasil["good_matches"])
    kp1, kp2 = hasil["jumlah_kp"]
    info = f"Good matches: {n_good} | Keypoints: {kp1} vs {kp2} | Ratio test: 0.75"
    cv2.putText(banner, info, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1)

    bagian = [baris1, baris2, gm, banner]

    # Strip ranking top-5 (mode identification)
    if ranking:
        strip = np.zeros((30 + 22 * min(5, len(ranking)), lebar, 3), np.uint8)
        cv2.putText(strip, "RANKING KANDIDAT DATABASE:", (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for i, r in enumerate(ranking[:5]):
            warna_r = (0, 200, 0) if i == 0 and cocok else (180, 180, 180)
            cv2.putText(strip, f"#{i+1}  {r['nama']:20s} skor {r['skor']:5.1f}%",
                        (15, 44 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_r, 1)
        bagian.append(strip)

    return np.vstack(bagian)


def cek_gui():
    """Deteksi OpenCV headless (tanpa dukungan GUI) dan beri pesan ramah.

    Penyebab umum: paket opencv-python-headless terinstall bersamaan
    dengan opencv-python, sehingga file cv2 versi headless menimpa yang GUI.
    Solusi: pip uninstall -y opencv-python opencv-python-headless
            pip install opencv-python
    """
    info = cv2.getBuildInformation()
    if not any(k in info for k in ("GTK", "Qt", "Cocoa", "Win32")):
        print("=" * 60)
        print("ERROR: OpenCV terinstall versi HEADLESS (tanpa GUI).")
        print("Jalankan perintah berikut lalu coba lagi:")
        print("  pip uninstall -y opencv-python opencv-python-headless")
        print("  pip install opencv-python")
        print("=" * 60)
        sys.exit(1)


def utama():
    cek_gui()
    ap = argparse.ArgumentParser(description="Fingerprint Matcher — UAS CV")
    ap.add_argument("query", nargs="?", help="citra query (jari yang dites)")
    ap.add_argument("referensi", nargs="?", help="citra referensi (mode verifikasi)")
    ap.add_argument("--db", default="database/", help="folder database pengguna")
    args = ap.parse_args()

    path_q = args.query or "samples/jari_A.png"
    path_r = args.referensi or "samples/jari_A_rotasi.png"
    mode = "verifikasi"          # "verifikasi" (1-lawan-1) / "identifikasi" (1-lawan-N)

    cv2.namedWindow(JENDELA, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold %", JENDELA, 7, 100, lambda x: None)
    cv2.createTrackbar("ClipLimit x10", JENDELA, 30, 80, lambda x: None)
    cv2.createTrackbar("Unsharp x10", JENDELA, 0, 30, lambda x: None)

    print("Kontrol: 'o'=query, 'r'=referensi, 'i'=identifikasi (cari pemilik),")
    print("         'v'=kembali ke verifikasi, 's'=simpan, 'q'=keluar")
    terakhir = None
    panel = None

    while True:
        threshold = cv2.getTrackbarPos("Threshold %", JENDELA)
        clip = max(1, cv2.getTrackbarPos("ClipLimit x10", JENDELA)) / 10.0
        unsharp = cv2.getTrackbarPos("Unsharp x10", JENDELA) / 10.0

        # Hitung ulang hanya kalau ada perubahan (hemat CPU)
        kondisi = (mode, path_q, path_r, threshold, clip, unsharp)
        if kondisi != terakhir:
            try:
                if mode == "identifikasi":
                    idn = identifikasi(path_q, args.db, threshold,
                                       clip_limit=clip, unsharp_k=unsharp)
                    terbaik = idn["terbaik"]
                    if terbaik is None:
                        print(f"Database kosong: {args.db}")
                        print("Jalankan dulu: python3 generate_samples.py")
                        break
                    status = (f"PEMILIK: {terbaik['nama'].upper()} / AKSES DITERIMA"
                              if idn["dikenal"] else
                              f"TIDAK DIKENAL (terdekat: {terbaik['nama']}) / AKSES DITOLAK")
                    r_tahap = terbaik["tahap"]
                    r_seg = segmentasi_lengkap(r_tahap["enhanced"])
                    panel = susun_panel(
                        idn["query_tahap"], r_tahap,
                        segmentasi_lengkap(idn["query_tahap"]["enhanced"]), r_seg,
                        terbaik["hasil"], threshold,
                        judul_ref=f"1. Kandidat: {terbaik['nama']}",
                        status_teks=status,
                        ranking=idn["ranking"],
                    )
                    print(f"[1-lawan-N] {len(idn['ranking'])} kandidat diperiksa -> "
                          f"{'Dikenal: ' + terbaik['nama'] if idn['dikenal'] else 'Tidak dikenal'}"
                          f" (skor terbaik {terbaik['skor']:.1f}%)")
                else:
                    q = load_grayscale(path_q)
                    r = load_grayscale(path_r)
                    q_tahap = enhance(q, clip_limit=clip, unsharp_k=unsharp)
                    r_tahap = enhance(r, clip_limit=clip, unsharp_k=unsharp)
                    q_seg = segmentasi_lengkap(q_tahap["enhanced"])
                    r_seg = segmentasi_lengkap(r_tahap["enhanced"])
                    hasil = cocokkan(q_tahap["enhanced"], r_tahap["enhanced"])
                    panel = susun_panel(q_tahap, r_tahap, q_seg, r_seg,
                                        hasil, threshold)
                    print(f"Skor: {hasil['skor']:.1f}% -> "
                          f"{'COCOK' if keputusan(hasil['skor'], threshold) else 'TIDAK COCOK'}")
            except FileNotFoundError as e:
                print(e)
                print("Jalankan dulu: python3 generate_samples.py")
                break
            terakhir = kondisi

        cv2.imshow(JENDELA, panel)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            p = pilih_file()
            if p:
                path_q = p
        elif key == ord('r'):
            p = pilih_file()
            if p:
                path_r = p
                mode = "verifikasi"
        elif key == ord('i'):
            p = pilih_file()
            if p:
                path_q = p
                mode = "identifikasi"
        elif key == ord('v'):
            mode = "verifikasi"
        elif key == ord('s'):
            cv2.imwrite("hasil_demo.png", panel)
            print("Panel disimpan sebagai hasil_demo.png")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    utama()