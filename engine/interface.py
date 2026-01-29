from typing import Dict, Any
from datetime import datetime, timezone
import subprocess
import sys
import os


def now_utc():
    return datetime.now(timezone.utc).isoformat()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def _run_py(relative_path: str):
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.exists(full_path):
        raise RuntimeError(f"파일 없음: {relative_path}")

    subprocess.run(
        [sys.executable, full_path],
        check=True,
    )


def run_pipeline(market: str = "KR") -> Dict[str, Any]:
    started_at = now_utc()

    try:
        _run_py("engine/free_history_collector/free_history_collector.py")
        _run_py("engine/normalize_history/normalize_history.py")
        _run_py("engine/signals_history_builder/signals_history_builder.py")
        _run_py("engine/strategy_research/strategy_research.py")

        return {
            "engine_mode": "engine.interface",
            "status": "DONE",
            "market": market,
            "started_at": started_at,
            "finished_at": now_utc(),
            "message": "Pipeline completed successfully",
        }

    except Exception as e:
        return {
            "engine_mode": "engine.interface",
            "status": "ERROR",
            "market": market,
            "started_at": started_at,
            "finished_at": now_utc(),
            "error": str(e),
        }


def recommend_top20():
    return {
        "engine_mode": "engine.interface",
        "generated_at": now_utc(),
        "note": "signals_latest.json 연결 예정",
        "items": [],
    }


def recommend_highrisk5():
    return {
        "engine_mode": "engine.interface",
        "generated_at": now_utc(),
        "items": [],
    }


def analyze_ticker(ticker: str):
    return {
        "engine_mode": "engine.interface",
        "generated_at": now_utc(),
        "ticker": ticker,
    }
