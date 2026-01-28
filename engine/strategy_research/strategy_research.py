# -*- coding: utf-8 -*-
"""
strategy_research.py (FINAL)
- Input:
  data/signals/signals_latest.json
  data/history_norm/KR/*.csv
  data/history_norm/US/*.csv
- Output:
  data/strategy/strategy_rankings_YYYYMMDD_HHMMSS.csv
  data/strategy/strategy_report_YYYYMMDD_HHMMSS.txt
  data/strategy/strategy_latest.json
"""
import os
import sys

def BASE_DIR():
    """
    ✅ Render/GitHub 서버 기준으로 data/ 경로가 항상 '레포 루트'에 생기도록 통일
    - 현재 파일 위치: engine/strategy_research/4.strategy_research.py (또는 strategy_research.py)
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
import traceback
from datetime import datetime

try:
    import pandas as pd
except Exception as e:
    raise SystemExit(
        "pandas가 필요합니다.\n"
        "설치: python -m pip install pandas\n"
        f"원인: {e}"
    )

# ✅ dist(또는 py) 기준으로 경로 통일 (레포 루트 기준)
HIST_DIR = P("data", "history_norm")
KR_DIR = P("data", "history_norm", "KR")
US_DIR = P("data", "history_norm", "US")

SIGNALS_DIR = P("data", "signals")
SIGNALS_LATEST = P("data", "signals", "signals_latest.json")

OUT_DIR = P("data", "strategy")
os.makedirs(OUT_DIR, exist_ok=True)

# === 백테스트 기본값(현호님 요청에 맞춰 “바로 실행 가능한” 현실적 값) ===
LOOKBACK = 900          # 최근 900봉(약 3.5년)
FEE_BPS = 8             # 왕복 수수료/슬리피지 합산 느낌(0.08%)
MAX_POS_PER_SYMBOL = 1  # 심플: 종목당 1포지션(롱)만

# 전략 3개 (기본)
STRATEGIES = ["MOM_TREND", "MEAN_REVERT", "BREAKOUT"]


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def read_csv_norm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(n): return cols.get(n.lower())
    c_date = pick("date")
    c_open = pick("open")
    c_high = pick("high")
    c_low = pick("low")
    c_close = pick("close")
    c_adj = pick("adjclose")

    need = [c_date, c_open, c_high, c_low, c_close]
    if any(x is None for x in need):
        raise ValueError(f"필수 컬럼 누락: {os.path.basename(path)} cols={list(df.columns)}")

    out = pd.DataFrame({
        "Date": df[c_date],
        "Open": pd.to_numeric(df[c_open], errors="coerce"),
        "High": pd.to_numeric(df[c_high], errors="coerce"),
        "Low": pd.to_numeric(df[c_low], errors="coerce"),
        "Close": pd.to_numeric(df[c_close], errors="coerce"),
    })
    if c_adj is not None:
        out["AdjClose"] = pd.to_numeric(df[c_adj], errors="coerce")
    else:
        out["AdjClose"] = out["Close"]

    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close", "AdjClose"])
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    return out


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["AdjClose"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def apply_fee(ret: float, fee_bps: float) -> float:
    # ret는 하루 수익률, fee는 거래 발생일에 차감
    return ret


def backtest_one(df: pd.DataFrame, strategy: str, fee_bps: float) -> dict:
    """
    Long-only, 1 position max.
    entry/exit 룰:
    - MOM_TREND: 60일 모멘텀>0, 20EMA>60EMA, RSI 40~75 진입 / 20EMA<60EMA 또는 RSI<35 이탈
    - MEAN_REVERT: RSI<30 진입 / RSI>55 이탈 (손절은 ATR 기반 간단 적용)
    - BREAKOUT: 20일 고가 돌파 진입 / 10일 저가 이탈 청산
    """
    df = df.copy()
    if len(df) < 260:
        return {"ok": False}

    if len(df) > LOOKBACK:
        df = df.iloc[-LOOKBACK:].copy()

    close = df["AdjClose"].astype(float)
    df["ret1"] = close.pct_change().fillna(0.0)

    df["ema20"] = ema(close, 20)
    df["ema60"] = ema(close, 60)
    df["rsi14"] = rsi(close, 14).fillna(50.0)
    df["atr14"] = atr(df, 14).fillna(0.0)

    df["hh20"] = df["High"].rolling(20).max()
    df["ll10"] = df["Low"].rolling(10).min()

    # 60일 모멘텀
    df["mom60"] = (close / close.shift(60)) - 1.0
    df["mom60"] = df["mom60"].fillna(0.0)

    pos = 0  # 0 or 1
    entry_price = 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = 0
    wins = 0

    for i in range(1, len(df)):
        price = float(df["AdjClose"].iloc[i])
        r = float(df["ret1"].iloc[i])
        rsi14v = float(df["rsi14"].iloc[i])
        ema20v = float(df["ema20"].iloc[i])
        ema60v = float(df["ema60"].iloc[i])
        atr14v = float(df["atr14"].iloc[i])
        hh20 = float(df["hh20"].iloc[i])
        ll10 = float(df["ll10"].iloc[i])
        mom60 = float(df["mom60"].iloc[i])

        # === EXIT 먼저 ===
        if pos == 1:
            exit_sig = False

            if strategy == "MOM_TREND":
                if (ema20v < ema60v) or (rsi14v < 35):
                    exit_sig = True

            elif strategy == "MEAN_REVERT":
                if rsi14v > 55:
                    exit_sig = True
                # 간단 손절: entry 대비 -2*ATR
                if atr14v > 0 and price < (entry_price - 2.0 * atr14v):
                    exit_sig = True

            elif strategy == "BREAKOUT":
                if price < ll10:
                    exit_sig = True

            if exit_sig:
                # 청산 수수료(왕복의 절반처럼 단순 차감)
                equity *= (1.0 - fee_bps / 10000.0)
                # 승패 판정
                if price > entry_price:
                    wins += 1
                pos = 0
                entry_price = 0.0

        # === ENTRY ===
        if pos == 0:
            enter_sig = False

            if strategy == "MOM_TREND":
                if (mom60 > 0) and (ema20v > ema60v) and (40 <= rsi14v <= 75):
                    enter_sig = True

            elif strategy == "MEAN_REVERT":
                if rsi14v < 30:
                    enter_sig = True

            elif strategy == "BREAKOUT":
                if price >= hh20 and hh20 > 0:
                    enter_sig = True

            if enter_sig:
                pos = 1
                entry_price = price
                trades += 1
                # 진입 수수료
                equity *= (1.0 - fee_bps / 10000.0)

        # === PnL 적용(보유 중이면 수익률 반영) ===
        if pos == 1:
            equity *= (1.0 + r)

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # 마지막 보유분 청산(수수료만 반영)
    if pos == 1:
        equity *= (1.0 - fee_bps / 10000.0)

    # 연환산 대충(거래일 252 기준, 기간 길이로 환산)
    days = max(1, len(df))
    cagr = (equity ** (252.0 / days)) - 1.0 if equity > 0 else -1.0
    winrate = (wins / trades) if trades > 0 else 0.0

    return {
        "ok": True,
        "equity": equity,
        "cagr": cagr,
        "max_dd": max_dd,
        "trades": trades,
        "winrate": winrate,
    }


def load_candidates() -> tuple[list[str], list[str]]:
    if not os.path.exists(SIGNALS_LATEST):
        raise FileNotFoundError(
            f"signals_latest.json이 없습니다: {SIGNALS_LATEST}\n"
            "먼저 research.py(또는 signals_latest.json 생성 단계)를 실행하세요."
        )

    with open(SIGNALS_LATEST, "r", encoding="utf-8") as f:
        s = json.load(f)

    top_kr = s.get("top_kr", []) or []
    top_us = s.get("top_us", []) or []

    kr_syms = [x.get("symbol") for x in top_kr if x.get("symbol")]
    us_syms = [x.get("symbol") for x in top_us if x.get("symbol")]

    return kr_syms, us_syms


def symbol_path(market: str, sym: str) -> str:
    if market == "KR":
        return os.path.join(KR_DIR, f"{sym}.csv")
    return os.path.join(US_DIR, f"{sym}.csv")


def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    kr_syms, us_syms = load_candidates()
    log(f"Strategy Research 시작: KR_candidates={len(kr_syms)} US_candidates={len(us_syms)}")
    log(f"백테스트 옵션: lookback={LOOKBACK} fee_bps={FEE_BPS}")

    # 즉시 “작동 확인용” 파일부터 만들어 둠 (멈춘 것처럼 보이는 문제 방지)
    bootstrap = {
        "generated_at": now,
        "status": "RUNNING",
        "note": "This file is created immediately to prove execution is alive.",
        "results": []
    }
    latest_json = os.path.join(OUT_DIR, "strategy_latest.json")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(bootstrap, f, ensure_ascii=False, indent=2)

    rows = []
    total_jobs = (len(kr_syms) + len(us_syms)) * len(STRATEGIES)
    done = 0

    def run_market(market: str, syms: list[str]):
        nonlocal done
        for sym in syms:
            p = symbol_path(market, sym)
            if not os.path.exists(p):
                continue
            try:
                df = read_csv_norm(p)
            except Exception:
                continue

            for st in STRATEGIES:
                try:
                    r = backtest_one(df, st, FEE_BPS)
                    done += 1
                    if (done % 50 == 0) or (done == total_jobs):
                        log(f"[PROGRESS] {done}/{total_jobs}")

                    if not r.get("ok"):
                        continue

                    rows.append({
                        "market": market,
                        "symbol": sym,
                        "strategy": st,
                        "equity": r["equity"],
                        "cagr": r["cagr"],
                        "max_dd": r["max_dd"],
                        "trades": r["trades"],
                        "winrate": r["winrate"],
                    })
                except Exception:
                    done += 1
                    continue

    run_market("KR", kr_syms)
    run_market("US", us_syms)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        df_out = pd.DataFrame(columns=["market","symbol","strategy","equity","cagr","max_dd","trades","winrate"])

    # 성과 기준: CAGR - (maxDD*0.8) + (winrate*0.1)
    df_out["score"] = df_out["cagr"] - (df_out["max_dd"] * 0.8) + (df_out["winrate"] * 0.1)
    df_out = df_out.sort_values(["score"], ascending=False)

    csv_path = os.path.join(OUT_DIR, f"strategy_rankings_{now}.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    top5 = df_out.head(5).to_dict(orient="records") if not df_out.empty else []

    report_path = os.path.join(OUT_DIR, f"strategy_report_{now}.txt")
    rep = []
    rep.append(f"Strategy Research Report {now}")
    rep.append("=" * 70)
    rep.append(f"- candidates KR={len(kr_syms)} US={len(us_syms)}")
    rep.append(f"- strategies={STRATEGIES}")
    rep.append(f"- total_results={len(df_out)}")
    rep.append("")
    rep.append("[TOP 5 (overall)]")
    if top5:
        for r in top5:
            rep.append(
                f"{r['market']} {r['symbol']} {r['strategy']} "
                f"score={r['score']:+.4f} cagr={r['cagr']:+.4f} dd={r['max_dd']:.3f} "
                f"trades={int(r['trades'])} win={r['winrate']:.2f}"
            )
    else:
        rep.append("(empty)")

    rep.append("")
    rep.append("Output Files:")
    rep.append(f"- {csv_path}")
    rep.append(f"- {latest_json}")
    rep.append(f"- {report_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    latest_payload = {
        "generated_at": now,
        "status": "DONE",
        "top5": top5,
        "csv": os.path.basename(csv_path),
        "report": os.path.basename(report_path),
    }
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(latest_payload, f, ensure_ascii=False, indent=2)

    log(f"[WRITE] CSV    -> {csv_path}")
    log(f"[WRITE] LATEST -> {latest_json}")
    log(f"[WRITE] REPORT -> {report_path}")
    log("Strategy Research 완료")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        log(traceback.format_exc())
        raise
