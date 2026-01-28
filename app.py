from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# ------------------------------------------------------------
# Engine import (ONLY engine.interface)
#  - research.py fallback 제거 (import 충돌 방지)
# ------------------------------------------------------------
ENGINE_MODE = "dummy"

recommend_top20 = None
recommend_highrisk5 = None
analyze_ticker = None
run_pipeline = None

try:
    # engine/interface.py 안에 아래 함수들이 있어야 정상 동작
    from engine.interface import (
        recommend_top20,
        recommend_highrisk5,
        analyze_ticker,
        run_pipeline,
    )  # type: ignore

    ENGINE_MODE = "engine.interface"
except Exception:
    ENGINE_MODE = "dummy"


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(
    title="Quant Recommend API",
    version="0.1.0",
    description="Stock 추천/분석 API (engine.interface 연결)",
)

# CORS (웹/앱에서 호출 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 나중에 앱 도메인 정해지면 여기 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------
# Dummy implementations (engine.interface 없을 때 대비)
# ------------------------------------------------------------
def dummy_top20() -> Dict[str, Any]:
    items = []
    for i in range(1, 21):
        items.append(
            {
                "rank": i,
                "ticker": f"00000{i % 10}",
                "name": f"DummyStock{i}",
                "score": round(100 - i * 2.3, 2),
                "prob": round(0.55 + (20 - i) * 0.01, 2),
                "grade": "B+",
            }
        )
    return {
        "engine_mode": ENGINE_MODE,
        "generated_at": now_utc_iso(),
        "count": len(items),
        "items": items,
    }


def dummy_highrisk5() -> Dict[str, Any]:
    items = []
    for i in range(1, 6):
        items.append(
            {
                "rank": i,
                "ticker": f"HR{i:03d}",
                "name": f"HighRisk{i}",
                "score": round(80 - i * 1.7, 2),
                "prob": round(0.50 + (5 - i) * 0.03, 2),
                "grade": "A-",
                "note": "고위험/고수익 후보",
            }
        )
    return {
        "engine_mode": ENGINE_MODE,
        "generated_at": now_utc_iso(),
        "count": len(items),
        "items": items,
    }


def dummy_analyze(ticker: str) -> Dict[str, Any]:
    return {
        "engine_mode": ENGINE_MODE,
        "generated_at": now_utc_iso(),
        "ticker": ticker,
        "summary": "더미 분석 결과입니다 (engine 연결 전).",
        "scenarios": [
            {"name": "상승", "prob": 0.45, "plan": "분할매수 후 목표가 분할매도"},
            {"name": "횡보", "prob": 0.35, "plan": "박스권 하단 재매수, 상단 매도"},
            {"name": "하락", "prob": 0.20, "plan": "손절/비중축소, 재진입 조건 대기"},
        ],
        "trade_plan": {
            "entry": None,
            "take_profit": None,
            "stop_loss": None,
        },
        "details": {},
    }


def dummy_run_pipeline(market: str = "KR") -> Dict[str, Any]:
    return {
        "engine_mode": ENGINE_MODE,
        "generated_at": now_utc_iso(),
        "status": "dummy_ok",
        "market": market,
        "message": "더미 파이프라인 실행 완료 (engine 연결 전).",
    }


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
@app.get("/", tags=["default"])
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "engine_mode": ENGINE_MODE,
        "generated_at": now_utc_iso(),
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["default"])
def health() -> Dict[str, Any]:
    return {"ok": True, "engine_mode": ENGINE_MODE, "generated_at": now_utc_iso()}


@app.get("/recommend/top20", tags=["recommend"])
def api_top20() -> Dict[str, Any]:
    # engine.interface가 있으면 진짜 함수 호출
    if recommend_top20:
        try:
            return recommend_top20()  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"recommend_top20 error: {e}")

    # 없으면 더미
    return dummy_top20()


@app.get("/recommend/highrisk", tags=["recommend"])
def api_highrisk() -> Dict[str, Any]:
    if recommend_highrisk5:
        try:
            return recommend_highrisk5()  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"recommend_highrisk5 error: {e}")

    return dummy_highrisk5()


@app.get("/analyze/{ticker}", tags=["analyze"])
def api_analyze(ticker: str) -> Dict[str, Any]:
    ticker = (ticker or "").strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    if analyze_ticker:
        try:
            return analyze_ticker(ticker)  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"analyze_ticker error: {e}")

    return dummy_analyze(ticker)


@app.post("/engine/run", tags=["engine"])
def api_engine_run(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    파이프라인(수집->정규화->리서치->시그널->전략)을 서버에서 실행시키는 용도
    payload 예시:
      {"market":"KR"}
    """
    market = "KR"
    if payload and isinstance(payload, dict):
        market = str(payload.get("market", "KR")).upper().strip() or "KR"

    if run_pipeline:
        try:
            return run_pipeline(market=market)  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"run_pipeline error: {e}")

    return dummy_run_pipeline(market=market)
