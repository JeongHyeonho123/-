from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# ------------------------------------------------------------
# Optional engine imports (works even if engine is not ready yet)
# ------------------------------------------------------------
ENGINE_MODE = "dummy"

hello_engine = None
recommend_top20 = None
recommend_highrisk5 = None
analyze_ticker = None

# 1) Try to use "engine/interface.py" if you create it later
try:
    from engine.interface import recommend_top20, recommend_highrisk5, analyze_ticker  # type: ignore
    ENGINE_MODE = "engine.interface"
except Exception:
    # 2) If not available, try the test function "engine/research.py"
    try:
        from engine.research import hello_engine  # type: ignore
        ENGINE_MODE = "engine.research(hello_engine)"
    except Exception:
        ENGINE_MODE = "dummy"


# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
app = FastAPI(title="Quant Recommend API", version="0.1.0")

# CORS: allow app/web clients to call this API
# (For production, you should restrict origins.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_kst_iso() -> str:
    # KST = UTC+9
    kst = timezone.utc
    # We'll return UTC time with offset "Z" by default; you can adjust later.
    # If you really want +09:00 formatting, we can change it later.
    return datetime.now(timezone.utc).isoformat()


def engine_status() -> str:
    if callable(hello_engine):
        try:
            return str(hello_engine())
        except Exception as e:
            return f"engine hello error: {e}"
    return f"engine mode: {ENGINE_MODE}"


# ------------------------------------------------------------
# Dummy implementations (used until engine is fully wired)
# ------------------------------------------------------------
def dummy_top20() -> Dict[str, Any]:
    return {
        "asof": now_kst_iso(),
        "mode": "eod",
        "horizon_days": 20,
        "engine": engine_status(),
        "items": [
            {
                "ticker": "005930",
                "name": "삼성전자",
                "grade": "B+",
                "prob": {"up": 0.63, "flat": 0.21, "down": 0.16},
                "score_bin": "80-85",
            },
            {
                "ticker": "000660",
                "name": "SK하이닉스",
                "grade": "B",
                "prob": {"up": 0.58, "flat": 0.24, "down": 0.18},
                "score_bin": "75-80",
            },
        ],
    }


def dummy_highrisk5() -> Dict[str, Any]:
    return {
        "asof": now_kst_iso(),
        "mode": "eod",
        "horizon_days": 20,
        "risk_tag": "high",
        "engine": engine_status(),
        "items": [
            {
                "ticker": "263750",
                "name": "펄어비스(예시)",
                "grade": "C+",
                "prob": {"up": 0.55, "flat": 0.15, "down": 0.30},
                "score_bin": "70-75",
                "note": "변동성 높음(예시)",
            }
        ],
    }


def dummy_analyze(ticker: str) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "name": None,
        "asof": now_kst_iso(),
        "grade": "B+",
        "prob": {"up": 0.63, "flat": 0.21, "down": 0.16},
        "n_samples": 0,
        "levels": {"entry": None, "stop": None, "tp1": None, "tp2": None},
        "reasons_top3": [
            {"factor": "trend", "note": "테스트 더미(엔진 연결 전)"},
            {"factor": "volume", "note": "테스트 더미(엔진 연결 전)"},
            {"factor": "momentum", "note": "테스트 더미(엔진 연결 전)"},
        ],
        "engine": engine_status(),
    }


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine_mode": ENGINE_MODE,
        "engine": engine_status(),
        "asof": now_kst_iso(),
    }


@app.get("/recommend/top20")
def api_top20() -> Dict[str, Any]:
    # If engine interface exists, use it
    if callable(recommend_top20):
        try:
            data = recommend_top20()  # expected dict JSON-serializable
            # Add metadata if missing
            if isinstance(data, dict):
                data.setdefault("asof", now_kst_iso())
                data.setdefault("engine", engine_status())
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"recommend_top20 error: {e}")

    # Fallback dummy
    return dummy_top20()


@app.get("/recommend/highrisk")
def api_highrisk() -> Dict[str, Any]:
    if callable(recommend_highrisk5):
        try:
            data = recommend_highrisk5()  # expected dict
            if isinstance(data, dict):
                data.setdefault("asof", now_kst_iso())
                data.setdefault("engine", engine_status())
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"recommend_highrisk5 error: {e}")

    return dummy_highrisk5()


@app.get("/analyze/{ticker}")
def api_analyze(ticker: str) -> Dict[str, Any]:
    ticker = ticker.strip()

    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is empty")

    if callable(analyze_ticker):
        try:
            data = analyze_ticker(ticker)  # expected dict
            if isinstance(data, dict):
                data.setdefault("ticker", ticker)
                data.setdefault("asof", now_kst_iso())
                data.setdefault("engine", engine_status())
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"analyze_ticker error: {e}")

    return dummy_analyze(ticker)

# ==============================
# 엔진 파이프라인 수동 실행용 API
# (최초 1회 캐시 생성)
# ==============================

from engine.interface import run_pipeline

@app.post("/engine/run")
def engine_run():
    result = run_pipeline()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result)
    return result
