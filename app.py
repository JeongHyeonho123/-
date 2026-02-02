from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from datetime import datetime, timezone

app = FastAPI(title="Quant Recommend API (FREE MODE)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 메모리 저장소 (서버 재시작 시 초기화됨)
LATEST_RESULTS: Dict[str, Any] = {}


def now():
    return datetime.now(timezone.utc).isoformat()


@app.get("/")
def root():
    return {"ok": True, "mode": "FREE", "time": now()}


# ----------------------------
# 1) 집 PC → 결과 업로드
# ----------------------------
@app.post("/upload/result")
def upload_result(payload: Dict[str, Any]):
    name = payload.get("name")
    data = payload.get("data")

    if not name or data is None:
        raise HTTPException(400, "name/data required")

    LATEST_RESULTS[name] = {
        "uploaded_at": now(),
        "data": data,
    }

    return {"ok": True, "saved": name}


# ----------------------------
# 2) 앱 → 결과 조회
# ----------------------------
@app.get("/result/{name}")
def get_result(name: str):
    if name not in LATEST_RESULTS:
        raise HTTPException(404, "no data yet")

    return {
        "ok": True,
        "name": name,
        **LATEST_RESULTS[name],
    }
