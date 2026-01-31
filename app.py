from __future__ import annotations

import os
import json
from uuid import uuid4
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------
# Storage path (Render Disk 우선)
#  - Render에서 Disk를 /var/data 로 마운트했다고 가정
#  - 없으면 로컬 ./data 사용
# ------------------------------------------------------------
PERSIST_ROOT = os.environ.get("PERSIST_ROOT", "/var/data")
if not os.path.isdir(PERSIST_ROOT):
    # 디스크가 없으면 프로젝트 내부 data로 fallback
    PERSIST_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATA_DIR = os.path.join(PERSIST_ROOT, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded")

JOBS_PENDING = os.path.join(DATA_DIR, "jobs", "pending")
JOBS_DONE = os.path.join(DATA_DIR, "jobs", "done")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(JOBS_PENDING, exist_ok=True)
os.makedirs(JOBS_DONE, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# ------------------------------------------------------------
# Engine import (ONLY engine.interface)
# ------------------------------------------------------------
ENGINE_MODE = "dummy"

recommend_top20 = None
recommend_highrisk5 = None
analyze_ticker = None
run_pipeline = None

try:
    from engine.interface import (  # type: ignore
        recommend_top20,
        recommend_highrisk5,
        analyze_ticker,
        run_pipeline,
    )
    ENGINE_MODE = "engine.interface"
except Exception:
    ENGINE_MODE = "dummy"


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(
    title="Quant Recommend API",
    version="0.2.0",
    description="Stock 추천/분석 API (PC연산 결과 업로드/조회 + optional jobs)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Dummy implementations
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
# Default/Health
# ------------------------------------------------------------
@app.get("/", tags=["default"])
def root() -> Dict[str, Any]:
    return {"ok": True, "engine_mode": ENGINE_MODE, "generated_at": now_utc_iso(), "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["default"])
def health() -> Dict[str, Any]:
    return {"ok": True, "engine_mode": ENGINE_MODE, "generated_at": now_utc_iso(), "persist_root": PERSIST_ROOT}


# ------------------------------------------------------------
# (옵션) engine.interface 직접 호출용
#  - Render에서 실제 파이프라인 돌릴 생각이면 사용
#  - 현호님 구조(집PC 연산)에서는 거의 안 씀
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


@app.post("/engine/run", tags=["engine"])
def api_engine_run(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    market = "KR"
    if payload and isinstance(payload, dict):
        market = str(payload.get("market", "KR")).upper().strip() or "KR"

    if run_pipeline:
        try:
            return run_pipeline(market=market)  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"run_pipeline error: {e}")

    return dummy_run_pipeline(market=market)


# ------------------------------------------------------------
# 핵심 1) 집PC -> 결과 업로드 (파일 업로드 방식)
#   업로드 파일명 예:
#     signals_latest.json
#     strategy_latest.json
# ------------------------------------------------------------
@app.post("/upload/result", tags=["upload"])
async def upload_result_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    return {"ok": True, "engine_mode": ENGINE_MODE, "saved": file.filename, "path": save_path}


# ------------------------------------------------------------
# 핵심 2) 앱 -> 결과 조회
# ------------------------------------------------------------
@app.get("/result/{name}", tags=["result"])
def get_result(name: str) -> Dict[str, Any]:
    path = os.path.join(UPLOAD_DIR, name)
    if not os.path.exists(path):
        return {"ok": False, "error": "not found", "name": name}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# (옵션) jobs: 앱 버튼 -> 작업요청 쌓기 / PC가 폴링해서 실행
#   2차 단계에서 사용
# ------------------------------------------------------------
@app.post("/jobs/request", tags=["jobs"])
def request_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    job_id = f"job_{now_str()}_{uuid4().hex[:6]}"
    job = {
        "job_id": job_id,
        "type": payload.get("type", "RECOMMEND"),
        "market": payload.get("market", "KR"),
        "requested_at": now_str(),
    }
    path = os.path.join(JOBS_PENDING, f"{job_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    return {"ok": True, "job_id": job_id}


@app.get("/jobs/next", tags=["jobs"])
def get_next_job() -> Dict[str, Any]:
    files = sorted(os.listdir(JOBS_PENDING))
    if not files:
        return {"ok": False, "message": "no job"}

    fname = files[0]
    src = os.path.join(JOBS_PENDING, fname)
    dst = os.path.join(JOBS_DONE, fname)
    os.rename(src, dst)

    with open(dst, "r", encoding="utf-8") as f:
        job = json.load(f)

    return {"ok": True, "job": job}
