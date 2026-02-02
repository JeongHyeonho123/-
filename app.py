from __future__ import annotations

import os
import json
from uuid import uuid4
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


# ------------------------------------------------------------
# Engine import (ONLY engine.interface)
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
    version="0.2.0",
    description="Stock 추천/분석 API + Jobs/Upload 결과 저장 (Render Disk 연동)",
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


def now_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# ------------------------------------------------------------
# Storage paths (✅ 단 한 번만 정의 / BASE_DIR 기준 통일)
#  - Render Disk Mount Path: /opt/render/project/src/data
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

JOBS_DIR = os.path.join(DATA_DIR, "jobs")
JOBS_PENDING = os.path.join(JOBS_DIR, "pending")
JOBS_DONE = os.path.join(JOBS_DIR, "done")

RESULTS_DIR = os.path.join(DATA_DIR, "results")
UPLOADED_DIR = os.path.join(DATA_DIR, "uploaded")

os.makedirs(JOBS_PENDING, exist_ok=True)
os.makedirs(JOBS_DONE, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADED_DIR, exist_ok=True)


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
        "trade_plan": {"entry": None, "take_profit": None, "stop_loss": None},
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
# Default routes
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
    return {
        "ok": True,
        "engine_mode": ENGINE_MODE,
        "generated_at": now_utc_iso(),
        "storage": {
            "DATA_DIR": DATA_DIR,
            "JOBS_PENDING": JOBS_PENDING,
            "JOBS_DONE": JOBS_DONE,
            "RESULTS_DIR": RESULTS_DIR,
            "UPLOADED_DIR": UPLOADED_DIR,
        },
    }


# ------------------------------------------------------------
# Recommend / Analyze
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
# Jobs API (앱 → 작업요청 / 집PC → 가져가기)
# ------------------------------------------------------------
@app.post("/jobs/request", tags=["jobs"])
def request_job(payload: Dict[str, Any]):
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

    return {"ok": True, "job_id": job_id, "saved_to": path}


@app.get("/jobs/next", tags=["jobs"])
def get_next_job():
    files = sorted([x for x in os.listdir(JOBS_PENDING) if x.lower().endswith(".json")])
    if not files:
        return {"ok": False, "message": "no job"}

    fname = files[0]
    src = os.path.join(JOBS_PENDING, fname)
    dst = os.path.join(JOBS_DONE, fname)

    os.rename(src, dst)

    with open(dst, "r", encoding="utf-8") as f:
        job = json.load(f)

    return {"ok": True, "job": job, "moved_to": dst}


# ------------------------------------------------------------
# Result upload/download (PC → 결과 업로드 / 앱 → 결과 조회)
# ------------------------------------------------------------
@app.post("/internal/upload_result", tags=["internal"])
def upload_result_json(payload: Dict[str, Any]):
    """
    JSON 형태로 업로드
    payload 예시:
      {"name":"signals_latest", "data":{...}}
    """
    name = payload.get("name")
    data = payload.get("data")

    if not name or data is None:
        raise HTTPException(status_code=400, detail="name/data required")

    safe_name = str(name).strip().replace("/", "_").replace("\\", "_")
    path = os.path.join(RESULTS_DIR, f"{safe_name}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"ok": True, "saved": safe_name, "path": path}


@app.post("/upload/result", tags=["upload"])
async def upload_result_file(file: UploadFile = File(...)):
    """
    파일 업로드 (multipart)
    - python-multipart 필요 (requirements.txt에 추가해야 함)
    """
    save_path = os.path.join(UPLOADED_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    return {
        "ok": True,
        "filename": file.filename,
        "path": save_path,
    }


@app.get("/result/{name}", tags=["result"])
def get_result(name: str):
    """
    /result/signals_latest.json
    처럼 호출하면 파일 내용을 JSON으로 반환
    """
    safe_name = name.replace("/", "_").replace("\\", "_")
    path1 = os.path.join(UPLOADED_DIR, safe_name)
    path2 = os.path.join(RESULTS_DIR, safe_name)

    path = path1 if os.path.exists(path1) else path2
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"not found: {safe_name}")

    # 업로드 파일이 JSON이 아닐 수도 있으니, 우선 JSON 시도 → 실패하면 텍스트로 반환
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return {"ok": True, "name": safe_name, "text": f.read()}
