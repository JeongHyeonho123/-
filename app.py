# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ------------------------------------------------------------
# API KEY (Render Environment Variables에서 설정한 값)
# ------------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "").strip()

def check_api_key(x_api_key: Optional[str]):
    if not API_KEY:
        # 운영에서 API_KEY 없으면 막는게 맞음
        raise HTTPException(status_code=500, detail="Server API_KEY is not set")
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ------------------------------------------------------------
# Engine import (engine.interface)
# ------------------------------------------------------------
ENGINE_MODE = "dummy"
recommend_top20 = None
recommend_highrisk5 = None
analyze_ticker = None
run_pipeline = None

try:
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
    description="Stock 추천/분석 API (engine.interface 연결 + 업로드 캐시)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Dummy (engine.interface 없을 때)
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
    return {"engine_mode": ENGINE_MODE, "generated_at": now_utc_iso(), "count": len(items), "items": items}

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
    return {"engine_mode": ENGINE_MODE, "generated_at": now_utc_iso(), "count": len(items), "items": items}

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
        "trade_plan": {"entry": None, "take_profit": None, "stop_loss": None},
        "details": {},
    }

# ------------------------------------------------------------
# 기본 API
# ------------------------------------------------------------
@app.get("/", tags=["default"])
def root() -> Dict[str, Any]:
    return {"ok": True, "engine_mode": ENGINE_MODE, "generated_at": now_utc_iso(), "docs": "/docs", "health": "/health"}

@app.get("/health", tags=["default"])
def health() -> Dict[str, Any]:
    return {"ok": True, "engine_mode": ENGINE_MODE, "generated_at": now_utc_iso()}

# ------------------------------------------------------------
# 추천/분석: 엔진이 있으면 엔진, 없으면 dummy
# ------------------------------------------------------------
@app.get("/recommend/top20", tags=["recommend"])
def api_top20() -> Dict[str, Any]:
    if recommend_top20:
        try:
            return recommend_top20()  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"recommend_top20 error: {e}")
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

# ------------------------------------------------------------
# (중요) 집PC -> Render로 결과 JSON 업로드 (API_KEY 필요)
# ------------------------------------------------------------
@app.post("/internal/upload_json", tags=["internal"])
def upload_json(payload: Dict[str, Any], x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """
    payload 예시:
    {
      "name": "signals_latest.json",
      "data": {...json...}
    }
    """
    check_api_key(x_api_key)

    name = (payload.get("name") or "").strip()
    data = payload.get("data")

    if not name or data is None:
        raise HTTPException(status_code=400, detail="name/data required")

    # 파일명 안전 처리(최소)
    name = name.replace("\\", "/").split("/")[-1]
    if not name.endswith(".json"):
        name += ".json"

    save_path = os.path.join(UPLOAD_DIR, name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "ok": True,
        "engine_mode": ENGINE_MODE,
        "saved": name,
        "path": save_path,
        "generated_at": now_utc_iso(),
    }

# ------------------------------------------------------------
# 앱/프론트 -> 결과 읽기
# ------------------------------------------------------------
@app.get("/result/{name}", tags=["result"])
def get_result(name: str) -> Dict[str, Any]:
    name = (name or "").strip().replace("\\", "/").split("/")[-1]
    path = os.path.join(UPLOAD_DIR, name)
    if not os.path.exists(path):
        return {"ok": False, "error": "not found", "name": name}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"ok": True, "name": name, "data": data, "served_at": now_utc_iso()}
