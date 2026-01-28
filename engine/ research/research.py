# -*- coding: utf-8 -*-
"""
research.py (FINAL)
- Input:
  data/history_norm/KR/*.csv
  data/history_norm/US/*.csv
  (CSV columns: Date,Open,High,Low,Close,Volume,AdjClose)
- Output:
  data/signals/rankings_YYYYMMDD_HHMMSS.csv
  data/signals/signals_YYYYMMDD_HHMMSS.json
  data/signals/signals_latest.json
  data/signals/report_YYYYMMDD_HHMMSS.txt
"""

import os, sys

def BASE_DIR():
    # exe 실행 시: dist 폴더
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # py 실행 시
    return os.path.dirname(os.path.abspath(__file__))

def P(*paths):
    return os.path.join(BASE_DIR(), *paths)


import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# pandas 있으면 사용, 없으면 에러 안내
try:
    import pandas as pd
except Exception as e:
    raise SystemExit(
        "pandas가 필요합니다.\n"
        "설치: python -m pip install pandas\n"
        f"원인: {e}"
    )

# ✅❗중요: 이 줄이 있으면 exe에서 _MEI(임시폴더)로 고정되어 전부 꼬입니다.
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))   <-- 삭제/비활성화해야 함

# ✅ dist(또는 py 파일 위치) 기준으로 경로 통일
HIST_DIR = P("data", "history_norm")
KR_DIR = P("data", "history_norm", "KR")
US_DIR = P("data", "history_norm", "US")

OUT_DIR = P("data", "signals")
os.makedirs(OUT_DIR, exist_ok=True)

# === 기본값(현호님 현재 데이터 규모 기준 안전하게) ===
LOOKBACK_BARS = 504      # 최근 2년(영업일)
MIN_ROWS = 260           # 최소 1년치 이상
WORKERS = 12

TOP_N = 40               # KR/US 각각 TOP 40을 signals에 담음

BUY_SCORE_TH = 1.30
SELL_SCORE_TH = -1.10


@dataclass
class Features:
    symbol: str
    market: str  # "KR" or "US"
    last_date: str
    last_close: float
    mom_3m: float
    mom_6m: float
    vol_ann: float
    rsi: float
    macd_hist: float
    atr: float
    score: float
    action: str


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return _safe_float(rsi.iloc[-1], 50.0)


def calc_macd_hist(close: pd.Series) -> float:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return _safe_float(hist.iloc[-1], 0.0)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return _safe_float(atr.iloc[-1], 0.0)


def calc_vol_ann(close: pd.Series) -> float:
    ret = close.pct_change().dropna()
    if len(ret) < 30:
        return 0.0
    vol = ret.std() * math.sqrt(252)
    return _safe_float(vol, 0.0)


def momentum(close: pd.Series, bars: int) -> float:
    if len(close) <= bars:
        return 0.0
    base = close.iloc[-bars-1]
    last = close.iloc[-1]
    if base <= 0:
        return 0.0
    return _safe_float((last / base) - 1.0, 0.0)


def score_formula(mom_3m, mom_6m, macd_hist, rsi, vol_ann, atr, last_close) -> float:
    # RSI 중앙(57.5) 부근 우대, 극단 감점
    rsi_center_bonus = 1.0 - abs((rsi - 57.5) / 57.5)
    rsi_penalty = 0.0
    if rsi >= 80:
        rsi_penalty = (rsi - 80) / 20.0
    elif rsi <= 20:
        rsi_penalty = (20 - rsi) / 20.0

    atr_pct = 0.0
    if last_close > 0:
        atr_pct = atr / last_close

    score = 0.0
    score += 2.2 * mom_6m
    score += 1.4 * mom_3m
    score += 0.25 * macd_hist
    score += 0.30 * rsi_center_bonus
    score -= 0.60 * min(vol_ann, 2.5)
    score -= 0.80 * min(atr_pct, 1.0)
    score -= 0.50 * rsi_penalty
    return _safe_float(score, 0.0)


def decide_action(score, mom_3m, mom_6m, macd_hist, rsi) -> str:
    if (score >= BUY_SCORE_TH) and (mom_3m > 0) and (mom_6m > 0) and (macd_hist > 0) and (40 <= rsi <= 75):
        return "BUY"
    if (score <= SELL_SCORE_TH) and (mom_3m < 0) and (mom_6m < 0) and (macd_hist < 0) and (rsi <= 45):
        return "SELL"
    return "HOLD"


def list_csv_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".csv")]


def load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 컬럼 방어(대소문/공백)
    cols = {c.strip().lower(): c for c in df.columns}
    def pick(name: str):
        return cols.get(name.lower())

    c_date = pick("date")
    c_open = pick("open")
    c_high = pick("high")
    c_low  = pick("low")
    c_close= pick("close")
    c_vol  = pick("volume")
    c_adj  = pick("adjclose")

    need = [c_date, c_open, c_high, c_low, c_close, c_vol]
    if any(x is None for x in need):
        raise ValueError(f"필수 컬럼 누락: {os.path.basename(path)} cols={list(df.columns)}")

    out = pd.DataFrame({
        "Date": df[c_date],
        "Open": pd.to_numeric(df[c_open], errors="coerce"),
        "High": pd.to_numeric(df[c_high], errors="coerce"),
        "Low": pd.to_numeric(df[c_low], errors="coerce"),
        "Close": pd.to_numeric(df[c_close], errors="coerce"),
        "Volume": pd.to_numeric(df[c_vol], errors="coerce"),
    })
    if c_adj is not None:
        out["AdjClose"] = pd.to_numeric(df[c_adj], errors="coerce")
    else:
        out["AdjClose"] = out["Close"]

    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    return out


def analyze_file(path: str, market: str):
    symbol = os.path.splitext(os.path.basename(path))[0]
    try:
        df = load_one_csv(path)
        if len(df) < MIN_ROWS:
            return None

        if len(df) > LOOKBACK_BARS:
            df = df.iloc[-LOOKBACK_BARS:].copy()

        close = df["AdjClose"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)

        last_close = _safe_float(close.iloc[-1], 0.0)
        last_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

        mom_3m = momentum(close, 63)
        mom_6m = momentum(close, 126)
        vol_ann = calc_vol_ann(close)
        rsi = calc_rsi(close, 14)
        macd_hist = calc_macd_hist(close)
        atr = calc_atr(high, low, close, 14)

        score = score_formula(mom_3m, mom_6m, macd_hist, rsi, vol_ann, atr, last_close)
        action = decide_action(score, mom_3m, mom_6m, macd_hist, rsi)

        return Features(
            symbol=symbol,
            market=market,
            last_date=last_date,
            last_close=last_close,
            mom_3m=mom_3m,
            mom_6m=mom_6m,
            vol_ann=vol_ann,
            rsi=rsi,
            macd_hist=macd_hist,
            atr=atr,
            score=score,
            action=action,
        )
    except Exception:
        return None


def _format_top_lines(df: pd.DataFrame, n: int) -> list[str]:
    if df.empty:
        return ["(empty)"]
    lines = []
    for _, r in df.head(n).iterrows():
        lines.append(
            f"{r['symbol']:>8s}  score={r['score']:+.4f} act={r['action']:<4s} "
            f"mom3={r['mom_3m']:+.3f} mom6={r['mom_6m']:+.3f} "
            f"rsi={r['rsi']:.1f} macdH={r['macd_hist']:+.4f} vol={r['vol_ann']:.3f}"
        )
    return lines


def write_outputs(all_feats: list[Features]):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    for f in all_feats:
        rows.append({
            "market": f.market,
            "symbol": f.symbol,
            "score": round(f.score, 6),
            "action": f.action,
            "mom_3m": round(f.mom_3m, 6),
            "mom_6m": round(f.mom_6m, 6),
            "vol_ann": round(f.vol_ann, 6),
            "rsi": round(f.rsi, 2),
            "macd_hist": round(f.macd_hist, 6),
            "atr": round(f.atr, 6),
            "last_close": round(f.last_close, 6),
            "last_date": f.last_date,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "market","symbol","score","action","mom_3m","mom_6m","vol_ann","rsi","macd_hist","atr","last_close","last_date"
        ])

    df = df.sort_values(["market", "score"], ascending=[True, False])

    rankings_path = os.path.join(OUT_DIR, f"rankings_{now}.csv")
    df.to_csv(rankings_path, index=False, encoding="utf-8-sig")

    top_kr = df[df["market"] == "KR"].head(TOP_N).copy()
    top_us = df[df["market"] == "US"].head(TOP_N).copy()

    signals = {
        "generated_at": now,
        "top_kr": top_kr.to_dict(orient="records"),
        "top_us": top_us.to_dict(orient="records"),
    }

    signals_path = os.path.join(OUT_DIR, f"signals_{now}.json")
    with open(signals_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)

    latest_path = os.path.join(OUT_DIR, "signals_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append(f"Research Report {now}")
    report_lines.append("=" * 70)
    report_lines.append(f"- total_valid: {len(df)}")
    report_lines.append(f"- KR top{TOP_N}: {len(top_kr)}  (BUY {(top_kr['action']=='BUY').sum() if not top_kr.empty else 0})")
    report_lines.append(f"- US top{TOP_N}: {len(top_us)}  (BUY {(top_us['action']=='BUY').sum() if not top_us.empty else 0})")
    report_lines.append("")
    report_lines.append("[Top 10 KR]")
    report_lines.extend(_format_top_lines(top_kr, 10))
    report_lines.append("")
    report_lines.append("[Top 10 US]")
    report_lines.extend(_format_top_lines(top_us, 10))
    report_lines.append("")
    report_lines.append("Output Files:")
    report_lines.append(f"- {rankings_path}")
    report_lines.append(f"- {signals_path}")
    report_lines.append(f"- {latest_path}")

    report_path = os.path.join(OUT_DIR, f"report_{now}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log(f"[WRITE] rankings -> {rankings_path}")
    log(f"[WRITE] signals   -> {signals_path}")
    log(f"[WRITE] latest    -> {latest_path}")
    log(f"[WRITE] report    -> {report_path}")


def main():
    kr_files = list_csv_files(KR_DIR)
    us_files = list_csv_files(US_DIR)

    log(f"Research 시작: KR_files={len(kr_files)} / US_files={len(us_files)}")
    log(f"옵션: lookback_bars={LOOKBACK_BARS} min_rows={MIN_ROWS} workers={WORKERS}")

    jobs = [(p, "KR") for p in kr_files] + [(p, "US") for p in us_files]
    total = len(jobs)
    done = 0
    valid = 0
    feats: list[Features] = []

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        future_map = {ex.submit(analyze_file, p, m): (p, m) for p, m in jobs}
        for fut in as_completed(future_map):
            done += 1
            try:
                res = fut.result()
                if res is not None:
                    feats.append(res)
                    valid += 1
            except Exception:
                pass

            if done % 500 == 0 or done == total:
                log(f"[PROGRESS] {done}/{total} 완료, 유효결과={valid}")

    write_outputs(feats)
    log(f"Research 완료: valid={valid}")
    log("다음 단계: strategy_research.py 실행 → data/strategy/* 생성")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        log(traceback.format_exc())
