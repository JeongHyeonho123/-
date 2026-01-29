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
    """
    Render 서버에서 파이썬 스크립트를 순차 실행
    """
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.exists(full_path):
        raise RuntimeError(f"파일 없음: {relative_path}")

    subprocess.run(
        [sys.executable, full_path],
        check=True,
    )


# -------------------------------------------------
# API에서 호출되는 엔진 함수들
# -------------------------------------------------

def run_pipeline(market: str = "KR") -> Dict[str, Any]:
    """
    전체 퀀트 파이프라인 실행
    """
    started_at = now_utc()

    try:
        # 1️⃣ 히스토리 수집
        _run_py("engine/free_history_collector/free_history_collector.py")

        # 2️⃣ 정규화
        _run_py("engine/normalize_history/normalize_history.py")

        # 3️⃣ 시그널 생성
        _run_py("engine/signals_history_builder/signals_history_builder.py")

        # 4️⃣ 전략 리서치
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


# -------------------------------------------------
# 아직 연결 안 된 API용 (임시)
# -------------------------------------------------

def recommend_top20():
    return {
        "engine_mode": "engine.interface",
        "generated_at": now_utc(),
        "note": "pipeline 결과 연결 전 (signals_latest.json 사용 예정)",
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
        "note": "strategy/analysis 연결 예정",
    }
