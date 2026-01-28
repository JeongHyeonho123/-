# engine/interface.py
from __future__ import annotations
from typing import Any, Dict
from datetime import datetime, timezone

print("🔥 engine.interface LOADED OK")  # 로그 확인용

def _now():
    return datetime.now(timezone.utc).isoformat()

def recommend_top20() -> Dict[str, Any]:
    return {
        "engine_mode": "engine.interface",
        "generated_at": _now(),
        "items": [
            {"rank": i, "ticker": f"KR{i:04d}", "score": 100 - i}
            for i in range(1, 21)
        ],
    }

def recommend_highrisk5() -> Dict[str, Any]:
    return {
        "engine_mode": "engine.interface",
        "generated_at": _now(),
        "items": [
            {"rank": i, "ticker": f"HR{i:03d}", "score": 90 - i}
            for i in range(1, 6)
        ],
    }

def analyze_ticker(ticker: str) -> Dict[str, Any]:
    return {
        "engine_mode": "engine.interface",
        "generated_at": _now(),
        "ticker": ticker,
        "summary": "engine.interface 연결 성공",
    }

def run_pipeline(market: str = "KR") -> Dict[str, Any]:
    return {
        "engine_mode": "engine.interface",
        "generated_at": _now(),
        "status": "ok",
        "market": market,
    }
