# ============================================================
# evaluate.py — Uji akurasi batch (untuk tabel di laporan UAS)
#
# Mode 1 (sampel sintetis, langsung bisa jalan):
#   python3 evaluate.py --synthetic
#
# Mode 2 (dataset SOCOFing asli):
#   python3 evaluate.py --dir dataset/Real --pairs 100
#   Nama file SOCOFing: "1__M_Left_index_finger_CR.BMP"
#   -> identitas jari = bagian sebelum "__"
#
# Output: akurasi, FAR (false accept), FRR (false reject)
# FAR/FRR = bukti analisis trade-off threshold saat demo.
# ============================================================

import argparse
import glob
import os
import random

from preprocessing import load_grayscale, enhance
from matcher import cocokkan


def uji_pasangan(daftar_pasangan, threshold, tampilkan=True):
    """daftar_pasangan: list (path1, path2, label_sama: bool)"""
    benar = 0
    false_accept = 0   # jari BEDA tapi diterima (bahaya keamanan)
    false_reject = 0   # jari SAMA tapi ditolak (tidak nyaman)
    total_sama = total_beda = 0
    daftar_skor = []

    for p1, p2, sama in daftar_pasangan:
        q = enhance(load_grayscale(p1))["enhanced"]
        r = enhance(load_grayscale(p2))["enhanced"]
        skor = cocokkan(q, r)["skor"]
        daftar_skor.append((skor, sama))
        prediksi = skor >= threshold
        if prediksi == sama:
            benar += 1
        if sama:
            total_sama += 1
            if not prediksi:
                false_reject += 1
        else:
            total_beda += 1
            if prediksi:
                false_accept += 1
        if tampilkan:
            status = "COCOK" if prediksi else "TIDAK"
            print(f"  {os.path.basename(p1):28s} vs {os.path.basename(p2):28s} "
                  f"skor={skor:5.1f}%  {status:5s}  (seharusnya: {'SAMA' if sama else 'BEDA'})")

    n = len(daftar_pasangan)
    if tampilkan:
        print("\n=== HASIL EVALUASI ===")
        print(f"Threshold        : {threshold}%")
        print(f"Total pasangan   : {n} ({total_sama} sama, {total_beda} beda)")
        print(f"Akurasi          : {benar / n * 100:.1f}%")
        if total_beda:
            print(f"FAR (beda diterima): {false_accept / total_beda * 100:.1f}%")
        if total_sama:
            print(f"FRR (sama ditolak) : {false_reject / total_sama * 100:.1f}%")
    return daftar_skor


def sweep_threshold(daftar_skor):
    """[Untuk laporan] Tabel kalibrasi: akurasi / FAR / FRR per threshold.

    Menunjukkan trade-off keamanan vs kenyamanan secara kuantitatif.
    """
    print("\n=== KALIBRASI THRESHOLD (untuk tabel laporan) ===")
    print(f"{'Threshold':>9} | {'Akurasi':>7} | {'FAR':>6} | {'FRR':>6}")
    terbaik = (0, -1)
    for t in range(2, 31, 2):
        benar = fa = fr = ns = nb = 0
        for skor, sama in daftar_skor:
            prediksi = skor >= t
            if prediksi == sama:
                benar += 1
            if sama:
                ns += 1
                fr += not prediksi
            else:
                nb += 1
                fa += prediksi
        akurasi = benar / len(daftar_skor) * 100
        far = fa / nb * 100 if nb else 0
        frr = fr / ns * 100 if ns else 0
        print(f"{t:>8}% | {akurasi:>6.1f}% | {far:>5.1f}% | {frr:>5.1f}%")
        if akurasi > terbaik[1]:
            terbaik = (t, akurasi)
    print(f"\nThreshold terbaik untuk data ini: {terbaik[0]}% "
          f"(akurasi {terbaik[1]:.1f}%)")


def mode_synthetic(threshold, sweep=False):
    s = "samples/"
    pasangan = [
        (s + "jari_A.png", s + "jari_A.png", True),         # identik
        (s + "jari_A.png", s + "jari_A_rotasi.png", True),  # rotasi 15 derajat
        (s + "jari_A.png", s + "jari_A_noise.png", True),   # noisy (LS 4)
        (s + "jari_A.png", s + "jari_A_blur.png", True),    # blur
        (s + "jari_A.png", s + "jari_B.png", False),        # orang lain
        (s + "jari_A.png", s + "jari_C.png", False),        # orang lain
        (s + "jari_B.png", s + "jari_C.png", False),        # orang lain
    ]
    print("=== MODE SINTETIS ===")
    skor = uji_pasangan(pasangan, threshold, tampilkan=not sweep)
    if sweep:
        sweep_threshold(skor)


def mode_dataset(folder, n_pairs, threshold, sweep=False):
    files = sorted(glob.glob(os.path.join(folder, "*.*")))
    if not files:
        print(f"Tidak ada file di {folder}. Download SOCOFing dulu (lihat README).")
        return

    def identitas(path):
        return os.path.basename(path).split("__")[0]

    kelompok = {}
    for f in files:
        kelompok.setdefault(identitas(f), []).append(f)

    random.seed(42)
    pasangan = []
    ids = list(kelompok.keys())
    # Pasangan SAMA: dua file dari identitas yang sama
    multi = [i for i in ids if len(kelompok[i]) >= 2]
    for _ in range(n_pairs // 2):
        if not multi:
            break
        i = random.choice(multi)
        a, b = random.sample(kelompok[i], 2)
        pasangan.append((a, b, True))
    # Pasangan BEDA: dua identitas berbeda
    for _ in range(n_pairs - len(pasangan)):
        i, j = random.sample(ids, 2)
        pasangan.append((random.choice(kelompok[i]), random.choice(kelompok[j]), False))

    print(f"=== MODE DATASET: {folder} ({len(files)} file, {len(ids)} identitas) ===")
    skor = uji_pasangan(pasangan, threshold, tampilkan=not sweep)
    if sweep:
        sweep_threshold(skor)


def mode_identifikasi(threshold):
    """Uji mode 1-lawan-N: query harus menemukan pemiliknya di database/.

    Metrik:
      Rank-1 accuracy : jari terdaftar dikenali pemiliknya dengan benar
      False accept    : jari ASING dianggap salah satu pengguna
      Rank-1 benar + asing ditolak = akurasi identifikasi
    """
    from identification import identifikasi

    s = "samples/"
    uji = [
        (s + "jari_A.png", "Andi"),
        (s + "jari_A_rotasi.png", "Andi"),
        (s + "jari_A_noise.png", "Andi"),
        (s + "jari_A_blur.png", "Andi"),
        (s + "jari_B.png", "Budi"),
        (s + "jari_C.png", "Citra"),
        (s + "jari_asing.png", None),   # bukan pengguna -> harus ditolak
    ]
    print("=== MODE IDENTIFIKASI (1-lawan-N, database/) ===")
    benar = 0
    asing_lolos = 0
    n_asing = 0
    for query, harapan in uji:
        idn = identifikasi(query, "database/", threshold)
        terbaik = idn["terbaik"]
        terkenal = terbaik["nama"] if idn["dikenal"] else None
        cocok = terkenal == harapan
        if harapan:
            benar += cocok
        else:
            n_asing += 1
            asing_lolos += (terkenal is not None)
        status = f"dikenali sebagai {terkenal}" if terkenal else "TIDAK DIKENAL (ditolak)"
        tanda = "OK" if cocok else "GAGAL"
        print(f"  {os.path.basename(query):20s} -> {status:32s} "
              f"(skor {terbaik['skor']:5.1f}%)  {tanda}")

    n_dikenal = len(uji) - n_asing
    print("\n=== HASIL IDENTIFIKASI ===")
    print(f"Threshold        : {threshold}%")
    print(f"Rank-1 accuracy  : {benar}/{n_dikenal} = {benar / n_dikenal * 100:.1f}%")
    if n_asing:
        print(f"Jari asing lolos : {asing_lolos}/{n_asing}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="uji pakai sampel sintetis")
    ap.add_argument("--dir", help="folder dataset (mis. dataset/Real)")
    ap.add_argument("--pairs", type=int, default=100, help="jumlah pasangan uji")
    ap.add_argument("--threshold", type=float, default=7.0)
    ap.add_argument("--sweep", action="store_true",
                    help="tabel kalibrasi threshold (untuk laporan)")
    ap.add_argument("--identification", action="store_true",
                    help="uji mode 1-lawan-N: cari pemilik jari di database/")
    args = ap.parse_args()

    if args.identification:
        mode_identifikasi(args.threshold)
    elif args.dir:
        mode_dataset(args.dir, args.pairs, args.threshold, args.sweep)
    else:
        mode_synthetic(args.threshold, args.sweep)