# app.py (Render FREE 최적화: JSON 업로드/조회 + 메모리 저장)
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware


# =========================
# 기본 설정
# =========================
API_KEY = os.getenv("API_KEY", "").strip()  # Render 환경변수로 넣을 값
ENGINE_MODE = "relay_memory"               # 무료 플랜은 디스크 저장이 아니라 메모리 중계가 현실적


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_key(x_api_key: Optional[str]) -> None:
    """
    업로드 보호(선택이지만 기본값으로 켜둠)
    - Render 환경변수 API_KEY를 설정하면 보호됨
    - API_KEY를 비워두면 보호 OFF (권장X)
    """
    if not API_KEY:
        return
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# =========================
# 메모리 저장소 (무료 Render용)
# =========================
RESULT_STORE: Dict[str, Dict[str, Any]] = {}
# 구조:
# RESULT_STORE[name] = {"saved_at": "...", "data": {...}}


# =========================
# FastAPI
# =========================
app = FastAPI(
    title="Quant Recommend API (FREE relay)",
    version="1.0.0",
    description="집PC가 만든 결과(JSON)를 Render가 임시 보관하고 앱이 조회하는 구조 (무료 플랜 최적화)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 나중에 앱 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["default"])
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "engine_mode": ENGINE_MODE,
        "generated_at": now_utc_iso(),
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload_json": "POST /upload/{name}",
            "get_json": "GET /result/{name}",
            "recommend_top20": "GET /recommend/top20",
            "recommend_highrisk": "GET /recommend/highrisk",
        },
        "note": "무료 플랜은 디스크 영구 저장이 불가할 수 있어 메모리 저장소로 동작합니다. 서버 재시작 시 데이터는 사라질 수 있습니다.",
    }


@app.get("/health", tags=["default"])
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "engine_mode": ENGINE_MODE,
        "time": now_utc_iso(),
        "stored_keys": list(RESULT_STORE.keys()),
    }


# =========================
# 1) 집PC -> Render 결과 업로드 (JSON)
# =========================
@app.post("/upload/{name}", tags=["upload"])
def upload_result_json(
    name: str,
    payload: Dict[str, Any],
    x_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_key(x_api_key)

    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    RESULT_STORE[name] = {
        "saved_at": now_utc_iso(),
        "data": payload,
    }
    return {
        "ok": True,
        "engine_mode": ENGINE_MODE,
        "saved": name,
        "saved_at": RESULT_STORE[name]["saved_at"],
        "stored_count": len(RESULT_STORE),
    }


# =========================
# 2) 앱/폰 -> Render 결과 조회
# =========================
@app.get("/result/{name}", tags=["result"])
def get_result_json(name: str) -> Dict[str, Any]:
    name = (name or "").strip()
    if name not in RESULT_STORE:
        return {"ok": False, "error": "not found", "name": name, "engine_mode": ENGINE_MODE}

    return {
        "ok": True,
        "engine_mode": ENGINE_MODE,
        "name": name,
        "saved_at": RESULT_STORE[name]["saved_at"],
        "data": RESULT_STORE[name]["data"],
    }


# =========================
# 추천 API: 저장소에 올려둔 결과를 그대로 반환
# - 집PC가 아래 name으로 업로드하면 앱에서 바로 볼 수 있음
#   "recommend_top20"
#   "recommend_highrisk"
# =========================
@app.get("/recommend/top20", tags=["recommend"])
def api_top20() -> Dict[str, Any]:
    key = "recommend_top20"
    if key not in RESULT_STORE:
        return {
            "ok": False,
            "engine_mode": ENGINE_MODE,
            "error": f"{key} not uploaded yet",
            "hint": "집PC agen.py가 /upload/recommend_top20 으로 올리면 됩니다.",
        }
    return RESULT_STORE[key]["data"]


@app.get("/recommend/highrisk", tags=["recommend"])
def api_highrisk() -> Dict[str, Any]:
    key = "recommend_highrisk"
    if key not in RESULT_STORE:
        return {
            "ok": False,
            "engine_mode": ENGINE_MODE,
            "error": f"{key} not uploaded yet",
            "hint": "집PC agen.py가 /upload/recommend_highrisk 으로 올리면 됩니다.",
        }
    return RESULT_STORE[key]["data"]
