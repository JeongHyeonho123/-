# 2.normalize_history.py
# -*- coding: utf-8 -*-
"""
Normalize History (Phase 2) - 기업식 2차(MINIMAL)
목표:
- KR: 표준형(Date,Open,High,Low,Close,Volume,AdjClose)
- US: 멀티티커/단일티커 모두 표준형으로 저장

입력(기본):
  data/history_free/KR/*.csv
  data/history_free/US/*.csv
출력(기본):
  data/history_norm/KR/*.csv
  data/history_norm/US/*.csv

[2차 기업식 최소추가]
- exe/py 공통 경로 통일(ABS)
- 설정파일 기반(없으면 자동 생성)
- 로그 파일 기록
"""

import os, sys
def BASE_DIR():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def P(*paths):
    return os.path.join(BASE_DIR(), *paths)

def ABS(path: str) -> str:
    if not path:
        return path
    return path if os.path.isabs(path) else P(path)

import csv
import re
import json
from datetime import datetime

DEFAULT_CONFIG = {
    "in_kr_dir": "data/history_free/KR",
    "in_us_dir": "data/history_free/US",
    "out_kr_dir": "data/history_norm/KR",
    "out_us_dir": "data/history_norm/US",
    "min_rows": 200,
    "log_file": "data/history_norm/_normalize.log",
    "config_file": "normalize_history_config.json",
}

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class Logger:
    def __init__(self, path: str):
        self.path = ABS(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self, msg: str):
        line = f"[{now_ts()}] {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def read_json(path: str, default=None):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def merge_dict(base: dict, loaded: dict):
    for k, v in (loaded or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge_dict(base[k], v)
        else:
            base[k] = v

def load_config():
    cfg_path = ABS(DEFAULT_CONFIG["config_file"])
    if not os.path.exists(cfg_path):
        write_json(cfg_path, DEFAULT_CONFIG)
    loaded = read_json(cfg_path, default={}) or {}
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merge_dict(merged, loaded)

    for k in ("in_kr_dir", "in_us_dir", "out_kr_dir", "out_us_dir", "log_file", "config_file"):
        if isinstance(merged.get(k), str):
            merged[k] = ABS(merged[k])
    return merged

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_symbol(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    return s if s else "UNKNOWN"

def to_float(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None

def to_int(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        if "," in x:
            x = x.split(",")[0].strip()
        return int(float(x))
    except Exception:
        return None

def parse_date(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def write_standard_csv(out_path: str, rows: list):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "AdjClose"])
        w.writerows(rows)

def detect_kr_format(header: list) -> bool:
    h = [c.strip().lower() for c in header]
    return ("date" in h and "open" in h and "high" in h and "low" in h and "close" in h)

def normalize_kr_file(in_path: str, out_dir: str, min_rows: int):
    fname = os.path.basename(in_path)
    symbol = os.path.splitext(fname)[0]
    out_path = os.path.join(out_dir, f"{safe_symbol(symbol)}.csv")

    rows_out = []
    with open(in_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or not detect_kr_format(header):
            return 0, None

        idx = {name: i for i, name in enumerate(header)}
        for row in r:
            if not row or len(row) < 5:
                continue
            d = parse_date(row[idx["Date"]]) if "Date" in idx else parse_date(row[0])
            if not d:
                continue
            o = to_float(row[idx["Open"]]) if "Open" in idx else None
            h = to_float(row[idx["High"]]) if "High" in idx else None
            l = to_float(row[idx["Low"]])  if "Low"  in idx else None
            c = to_float(row[idx["Close"]]) if "Close" in idx else None
            v = to_int(row[idx["Volume"]]) if "Volume" in idx else None
            if c is None:
                continue
            rows_out.append([d, o, h, l, c, v, c])

    if len(rows_out) < min_rows:
        return len(rows_out), None

    rows_out.sort(key=lambda x: x[0])
    write_standard_csv(out_path, rows_out)
    return len(rows_out), out_path

def detect_us_multiticker_format(first_row: list, second_row: list) -> bool:
    if not first_row or not second_row:
        return False
    if len(first_row) < 10 or len(second_row) < 10:
        return False
    if (second_row[0] or "").strip() != "":
        return False
    if (first_row[0] or "").strip().lower() != "date":
        return False
    return True

def normalize_us_simple_file(in_path: str, out_dir: str, min_rows: int):
    fname = os.path.basename(in_path)
    symbol = os.path.splitext(fname)[0]
    out_path = os.path.join(out_dir, f"{safe_symbol(symbol)}.csv")

    rows_out = []
    with open(in_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return 0, None

        hnorm = [c.strip().lower() for c in header]
        if "date" not in hnorm:
            return 0, None

        def idx_of(name):
            try:
                return hnorm.index(name.lower())
            except Exception:
                return -1

        i_date = idx_of("date")
        i_open = idx_of("open")
        i_high = idx_of("high")
        i_low  = idx_of("low")
        i_close = idx_of("close")
        i_adj = idx_of("adj close")
        i_vol = idx_of("volume")

        for row in r:
            if not row:
                continue
            d = parse_date(row[i_date] if 0 <= i_date < len(row) else "")
            if not d:
                continue
            o = to_float(row[i_open]) if 0 <= i_open < len(row) else None
            h = to_float(row[i_high]) if 0 <= i_high < len(row) else None
            l = to_float(row[i_low])  if 0 <= i_low  < len(row) else None
            c = to_float(row[i_close]) if 0 <= i_close < len(row) else None
            v = to_int(row[i_vol]) if 0 <= i_vol < len(row) else None
            adj = to_float(row[i_adj]) if 0 <= i_adj < len(row) else None
            if c is None:
                continue
            if adj is None:
                adj = c
            rows_out.append([d, o, h, l, c, v, adj])

    if len(rows_out) < min_rows:
        return len(rows_out), None

    rows_out.sort(key=lambda x: x[0])
    write_standard_csv(out_path, rows_out)
    return len(rows_out), out_path

def normalize_us_file(in_path: str, out_dir: str, min_rows: int):
    created = 0
    with open(in_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        r = csv.reader(f)
        row1 = next(r, None)
        row2 = next(r, None)

        if not detect_us_multiticker_format(row1, row2):
            return normalize_us_simple_file(in_path, out_dir, min_rows=min_rows)

        colmap = {}
        for i in range(1, len(row1)):
            field = (row1[i] or "").strip()
            ticker = (row2[i] or "").strip()
            if not field or not ticker:
                continue
            colmap[(field.lower(), ticker)] = i

        tickers = sorted({t for (_, t) in colmap.keys()})
        data_by_ticker = {t: [] for t in tickers}

        for row in r:
            if not row:
                continue
            d = parse_date(row[0] if len(row) > 0 else "")
            if not d:
                continue

            for t in tickers:
                def _get(field):
                    i = colmap.get((field, t), -1)
                    return row[i] if 0 <= i < len(row) else None

                adj = to_float(_get("adj close"))
                c   = to_float(_get("close"))
                h   = to_float(_get("high"))
                l   = to_float(_get("low"))
                o   = to_float(_get("open"))
                v   = to_int(_get("volume"))

                if c is None:
                    continue
                if adj is None:
                    adj = c
                data_by_ticker[t].append([d, o, h, l, c, v, adj])

        for t, rows_out in data_by_ticker.items():
            if len(rows_out) < min_rows:
                continue
            rows_out.sort(key=lambda x: x[0])
            out_path = os.path.join(out_dir, f"{safe_symbol(t)}.csv")
            write_standard_csv(out_path, rows_out)
            created += 1

    return created, None

def main():
    cfg = load_config()
    logger = Logger(cfg["log_file"])

    IN_KR_DIR  = cfg["in_kr_dir"]
    IN_US_DIR  = cfg["in_us_dir"]
    OUT_KR_DIR = cfg["out_kr_dir"]
    OUT_US_DIR = cfg["out_us_dir"]
    MIN_ROWS = int(cfg.get("min_rows", 200))

    ensure_dir(OUT_KR_DIR)
    ensure_dir(OUT_US_DIR)

    kr_files = [os.path.join(IN_KR_DIR, x) for x in os.listdir(IN_KR_DIR)] if os.path.isdir(IN_KR_DIR) else []
    kr_files = [x for x in kr_files if x.lower().endswith(".csv")]
    us_files = [os.path.join(IN_US_DIR, x) for x in os.listdir(IN_US_DIR)] if os.path.isdir(IN_US_DIR) else []
    us_files = [x for x in us_files if x.lower().endswith(".csv")]

    logger.log(f"[START] BASE_DIR={BASE_DIR()}")
    logger.log(f"[START] KR_files={len(kr_files)}  US_files={len(us_files)} min_rows={MIN_ROWS}")
    logger.log(f"[PATH] IN_KR={IN_KR_DIR} IN_US={IN_US_DIR}")
    logger.log(f"[PATH] OUT_KR={OUT_KR_DIR} OUT_US={OUT_US_DIR}")

    ok_kr = 0
    ok_us = 0

    for i, fp in enumerate(kr_files, 1):
        n, outp = normalize_kr_file(fp, OUT_KR_DIR, min_rows=MIN_ROWS)
        if outp:
            ok_kr += 1
        if i % 500 == 0:
            logger.log(f"[KR] {i}/{len(kr_files)} ok={ok_kr}")

    for i, fp in enumerate(us_files, 1):
        created, outp = normalize_us_file(fp, OUT_US_DIR, min_rows=MIN_ROWS)
        ok_us += int(created)
        if i % 500 == 0:
            logger.log(f"[US] {i}/{len(us_files)} created_files={ok_us}")

    logger.log(f"[DONE] KR_normalized_files={ok_kr}  US_normalized_files_created={ok_us}")
    logger.log(f"-> output: {OUT_KR_DIR}")
    logger.log(f"-> output: {OUT_US_DIR}")

if __name__ == "__main__":
    main()
