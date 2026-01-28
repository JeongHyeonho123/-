# 12.signals_history_builder.py
# -*- coding: utf-8 -*-
"""
Signals History Builder (Phase 3) - 기업식 2차(MINIMAL)
- ✅ Render/GitHub 서버 기준 경로 통일(레포 루트)
- 로직/룰 변경 없음(입력 폴더를 history_norm으로 전환 + 설정파일화 + 로그)
- ✅ __main__ try/except 로 에러 원인 출력
"""
import os
import sys

def BASE_DIR():
    """
    ✅ Render/GitHub 서버 기준으로 data/ 경로가 항상 '레포 루트'에 생기도록 통일
    - 현재 파일 위치: engine/signals_history_builder/12.signals_history_builder.py
    - 레포 루트: 위로 2단계
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def P(*paths):
    return os.path.join(BASE_DIR(), *paths)

def ABS(path: str) -> str:
    if not path:
        return path
    return path if os.path.isabs(path) else P(path)

import json
import math
from datetime import datetime
from collections import defaultdict

import pandas as pd

DEFAULT_CONFIG = {
    "kr_dir": "data/history_norm/KR",          # ✅ 핵심: history_free -> history_norm
    "out_dir": "data/signals/history",
    "top_kr_per_day": 40,
    "min_bars": 80,
    "lookback_mom": 20,
    "lookback_vol": 20,
    "w_mom": 0.70,
    "w_vol": 0.30,
    "vol_spike_z": 0.5,
    "log_file": "data/signals/_signals_builder.log",
    "config_file": "signals_history_config.json",
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
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
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

    for k in ("kr_dir", "out_dir", "log_file", "config_file"):
        if isinstance(merged.get(k), str):
            merged[k] = ABS(merged[k])
    return merged

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def read_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns:
        raise ValueError(f"Date 컬럼 없음: {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"{col} 컬럼 없음: {path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close", "Volume"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def compute_features(df: pd.DataFrame, look_mom: int, look_vol: int) -> pd.DataFrame:
    df = df.copy()
    df["mom"] = df["Close"].pct_change(int(look_mom))
    vol_ma = df["Volume"].rolling(int(look_vol)).mean()
    vol_std = df["Volume"].rolling(int(look_vol)).std(ddof=0).replace(0, math.nan)
    df["vol_z"] = (df["Volume"] - vol_ma) / vol_std
    df["mom"] = df["mom"].replace([math.inf, -math.inf], math.nan)
    df["vol_z"] = df["vol_z"].replace([math.inf, -math.inf], math.nan)
    return df

def stock_code_from_filename(fn: str) -> str:
    base = os.path.splitext(os.path.basename(fn))[0]
    digits = "".join(ch for ch in base if ch.isdigit())
    return digits.zfill(6) if digits else base.strip()[:6]

def main():
    cfg = load_config()
    logger = Logger(cfg["log_file"])

    ROOT = BASE_DIR()
    KR_DIR = cfg["kr_dir"]
    OUT_DIR = cfg["out_dir"]

    TOP_KR_PER_DAY = int(cfg["top_kr_per_day"])
    MIN_BARS = int(cfg["min_bars"])
    LOOKBACK_MOM = int(cfg["lookback_mom"])
    LOOKBACK_VOL = int(cfg["lookback_vol"])
    W_MOM = float(cfg["w_mom"])
    W_VOL = float(cfg["w_vol"])
    VOL_SPIKE_Z = cfg.get("vol_spike_z", 0.5)

    logger.log(f"[INFO] ROOT={ROOT}")
    logger.log(f"[INFO] KR_DIR={KR_DIR}")
    logger.log(f"[INFO] OUT_DIR={OUT_DIR}")

    if not os.path.isdir(KR_DIR):
        logger.log(f"[ERROR] KR_DIR 없음: {KR_DIR}")
        return

    files = [os.path.join(KR_DIR, f) for f in os.listdir(KR_DIR) if f.lower().endswith(".csv")]
    if not files:
        logger.log(f"[ERROR] CSV 없음: {KR_DIR}")
        return

    daily_rows = defaultdict(list)
    ok_cnt, skip_cnt, err_cnt = 0, 0, 0
    total_candidates_before_filter = 0
    total_passed_filter = 0

    for path in files:
        try:
            code = stock_code_from_filename(path)
            df = read_ohlcv_csv(path)
            if len(df) < MIN_BARS:
                skip_cnt += 1
                continue

            df = compute_features(df, LOOKBACK_MOM, LOOKBACK_VOL)
            df2 = df.dropna(subset=["mom", "vol_z", "Close"]).copy()
            if df2.empty:
                skip_cnt += 1
                continue

            # iterrows -> itertuples (기존 유지)
            for r in df2.itertuples(index=False):
                d = pd.Timestamp(r.Date).strftime("%Y%m%d")
                mom = safe_float(r.mom, 0.0)
                volz = safe_float(r.vol_z, -999.0)
                close = safe_float(r.Close, 0.0)

                total_candidates_before_filter += 1
                if VOL_SPIKE_Z is not None and volz < float(VOL_SPIKE_Z):
                    continue
                total_passed_filter += 1

                score = W_MOM * mom + W_VOL * (volz / 10.0)

                daily_rows[d].append({
                    "symbol": code,
                    "action": "BUY",
                    "score": float(score),
                    "meta": {"mom": float(mom), "vol_z": float(volz), "close": float(close)}
                })

            ok_cnt += 1

        except Exception as e:
            err_cnt += 1
            logger.log(f"[ERR] {path} -> {e}")

    if not daily_rows:
        logger.log("[WARN] 생성된 signals가 0개입니다.")
        logger.log(f"       candidates_before_filter={total_candidates_before_filter}")
        logger.log(f"       passed_filter={total_passed_filter}")
        logger.log("       원인: VOL_SPIKE_Z 필터가 너무 강할 가능성이 큽니다.")
        logger.log("       해결: vol_spike_z=0.0 또는 -1.0 로 낮춰보세요.")
        return

    ensure_dir(OUT_DIR)
    dates = sorted(daily_rows.keys())
    total_days = 0
    total_written = 0

    for d in dates:
        rows = daily_rows[d]
        if not rows:
            continue
        rows = sorted(rows, key=lambda x: x.get("score", -9999), reverse=True)[:TOP_KR_PER_DAY]

        out_day_dir = os.path.join(OUT_DIR, d)
        ensure_dir(out_day_dir)

        out_path = os.path.join(out_day_dir, f"signals_{d}_000000.json")
        payload = {"generated_at": d + "_000000", "top_kr": rows, "top_us": []}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        total_days += 1
        total_written += len(rows)

    logger.log(f"[OK] signals history built. days={total_days} total_rows={total_written}")
    logger.log(f"     output={OUT_DIR}")
    logger.log(f"[INFO] files processed ok={ok_cnt} skip={skip_cnt} err={err_cnt}")
    logger.log(f"[INFO] candidates_before_filter={total_candidates_before_filter} passed_filter={total_passed_filter}")
    logger.log(f"[INFO] settings: TOP={TOP_KR_PER_DAY} MIN_BARS={MIN_BARS} MOM={LOOKBACK_MOM} VOL={LOOKBACK_VOL} VOL_SPIKE_Z={VOL_SPIKE_Z}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] signals_history_builder failed: {e}")
        import traceback
        print(traceback.format_exc())
        raise
