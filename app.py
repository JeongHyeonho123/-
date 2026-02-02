# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------
# Paths (ONE truth)
# ------------------------------------------------------------
def BASE_DIR() -> str:
    # Render: /opt/render/project/src
    return os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(BASE_DIR(), "data")
UPLOAD_DIR = os.path.join(DATA_ROOT, "uploaded")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------------------------------------
# Simple API KEY check
# ------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "")  # Render Environment에 넣은 값

def require_api_key(req: Request):
    if not API_KEY:
        # 키를 안 쓰는 모드(원하면 막아도 됨)
        return
    got = req.headers.get("x-api-key") or req.headers.get("X-API-KEY") or ""
    if got != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

# ------------------------------------------------------------
# Engine import (optional)
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
    )
    ENGINE_MODE = "engine.interface"
except Exception:
    ENGINE_MODE = "dummy"

# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------
app = FastAPI(
    title="Quant Recommend API",
    version="0.2.0",
    description="Upload results from home PC -> Read from phone/app",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ------------------------------------------------------------
# Default
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

# ------------------------------------------------------------
# Upload (JSON only)  ✅ 이걸 집PC agen.py가 호출
# ------------------------------------------------------------
@app.post("/internal/upload_json", tags=["internal"])
async def upload_json(req: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    require_api_key(req)

    name = str(payload.get("name") or "").strip()
    data = payload.get("data", None)

    if not name or data is None:
        raise HTTPException(status_code=400, detail="name/data required")

    # name sanitize (폴더 탈출 방지)
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
# Result fetch  ✅ 폰/앱/스웨거가 조회
# ------------------------------------------------------------
@app.get("/result/{name}", tags=["result"])
def get_result(name: str) -> Dict[str, Any]:
    name = (name or "").strip().replace("\\", "/").split("/")[-1]
    if not name.endswith(".json"):
        name += ".json"

    path = os.path.join(UPLOAD_DIR, name)
    if not os.path.exists(path):
        return {"ok": False, "error": "not found", "name": name}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {"ok": True, "name": name, "data": data}

# ------------------------------------------------------------
# (Optional) Recommend endpoint: 업로드된 signals_latest.json 기반으로 만들고 싶으면
# 일단은 result로 충분히 확인하고, 다음 단계에서 연결하자.
# ------------------------------------------------------------
