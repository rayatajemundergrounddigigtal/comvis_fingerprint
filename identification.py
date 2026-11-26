# ============================================================
# identification.py
# Mode 1-LAWAN-N (IDENTIFICATION): "jari ini milik siapa?"
#
# Beda dengan verification (1-lawan-1, matcher.py):
#   verification   : apakah query == referensi X?  (2 citra)
#   identification : siapa pemilik query di database? (1 citra + folder)
#
# Cara kerja: scan semua citra di folder database, cocokkan satu-satu
# dengan matcher.cocokkan(), urutkan skor, ambil kandidat terbaik.
# Nama file database = nama pemilik (mis. Andi.png).
#
# Catatan analisis (untuk laporan): makin besar database, makin besar
# peluang kebetulan cocok -> threshold identification sebaiknya LEBIH
# KETAT daripada verification.
# ============================================================

import glob
import os

from preprocessing import load_grayscale, enhance
from matcher import cocokkan

EKSTENSI = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")


def daftar_file_db(folder_db):
    files = []
    for ext in EKSTENSI:
        files.extend(glob.glob(os.path.join(folder_db, ext)))
    return sorted(files)


def identifikasi(query_path, folder_db="database/", threshold=8.0,
                 clip_limit=3.0, unsharp_k=0.0):
    """Scan database, kembalikan ranking kandidat + keputusan.

    Return dict:
      query_tahap : dict tahapan enhancement query (untuk panel)
      ranking     : list urut skor menurun, tiap item:
                    {nama, path, skor, tahap, hasil}
      terbaik     : item ranking[0] atau None
      dikenal     : bool (skor terbaik >= threshold)
    """
    q = load_grayscale(query_path)
    q_tahap = enhance(q, clip_limit=clip_limit, unsharp_k=unsharp_k)

    ranking = []
    for path in daftar_file_db(folder_db):
        db = load_grayscale(path)
        db_tahap = enhance(db, clip_limit=clip_limit, unsharp_k=unsharp_k)
        hasil = cocokkan(q_tahap["enhanced"], db_tahap["enhanced"])
        ranking.append({
            "nama": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "skor": hasil["skor"],
            "tahap": db_tahap,
            "hasil": hasil,
        })

    ranking.sort(key=lambda r: r["skor"], reverse=True)
    terbaik = ranking[0] if ranking else None
    dikenal = bool(terbaik and terbaik["skor"] >= threshold)

    return {
        "query_tahap": q_tahap,
        "ranking": ranking,
        "terbaik": terbaik,
        "dikenal": dikenal,
    }