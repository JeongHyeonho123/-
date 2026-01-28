# engine/interface.py
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import json
from datetime import datetime
from typing import Dict, Any, List

# ------------------------------------------------------------
# Base dir (repo root)
# ------------------------------------------------------------
def BASE_DIR():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def P(*paths):
    return os.path.join(BASE_DIR(), *paths)

# ------------------------------------------------------------
# Data paths (Render Disk 사용 시 /var/data 권장)
# ------------------------------------------------------------
DATA_ROOT = os.environ.get("DATA_ROOT", "/var/data")

HISTORY_FREE = os.path.join(DATA_ROOT, "history_free")
HISTORY_NORM = os.path.join(DATA_ROOT, "history_norm")
SIGNALS_DIR  = os.path.join(DATA_ROOT, "signals")
STRATEGY_DIR = os.path.join(DATA_ROOT, "strategy")

os.makedirs(DATA_ROOT, exist_ok=True)

# ------------------------------------------------------------
# Script paths
# ------------------------------------------------------------
COLLECTOR = P("engine", "free_history_collector", "free_history_collector.py")
NORMALIZE = P("engine", "normalize_history", "normalize_history.py")
RESEARCH  = P("engine", "research", "research.py")
SIGNALS   = P("engine", "signals_history_builder", "signals_history_builder.py")
STRATEGY  = P("engine", "strategy_research", "strategy_research.py")

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _run_script(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"script not found: {path}")

    proc = subprocess.run(
        [sys.executable, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"[ERROR] {path}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

def _now():
    return datetime.utcnow().isoformat()

# ------------------------------------------------------------
# Pipeline runner (POST /engine/run)
# ------------------------------------------------------------
def run_pipeline(market: str = "KR") -> Dict[str, Any]:
    """
    전체 파이프라인 실행
    """
    logs: List[str] = []
    def step(name: str, fn):
        logs.append(f"[START] {name}")
        fn()
        logs.append(f"[DONE]  {name}")

    step("free_history_collector", lambda: _run_script(COLLECTOR))
    step("normalize_history",      lambda: _run_script(NORMALIZE))
    step("research",               lambda: _run_script(RESEARCH))
    step("signals_history_builder",lambda: _run_script(SIGNALS))
    step("strategy_research",      lambda: _run_script(STRATEGY))

    return {
        "ok": True,
        "engine": "legacy_quant_pipeline",
        "market": market,
        "generated_at": _now(),
        "logs": logs,
    }

# ------------------------------------------------------------
# Recommendation APIs
# ------------------------------------------------------------
def _load_strategy_latest() -> Dict[str, Any]:
    path = os.path.join(STRATEGY_DIR, "strategy_latest.json")
    if not os.path.exists(path):
        raise FileNotFoundError("strategy_latest.json not found. Run /engine/run first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def recommend_top20() -> Dict[str, Any]:
    data = _load_strategy_latest()
    top = data.get("top5", [])  # 임시: 이후 20개로 확장 가능

    items = []
    for i, r in enumerate(top, 1):
        items.append({
            "rank": i,
            "ticker": r.get("symbol"),
            "strategy": r.get("strategy"),
            "score": round(r.get("score", 0), 4),
            "grade": "B+",
        })

    return {
        "generated_at": _now(),
        "count": len(items),
        "items": items,
    }

def recommend_highrisk5() -> Dict[str, Any]:
    data = _load_strategy_latest()
    top = data.get("top5", [])

    items = []
    for i, r in enumerate(top, 1):
        items.append({
            "rank": i,
            "ticker": r.get("symbol"),
            "strategy": r.get("strategy"),
            "risk": "HIGH",
            "score": round(r.get("score", 0), 4),
        })

    return {
        "generated_at": _now(),
        "count": len(items),
        "items": items,
    }

def analyze_ticker(ticker: str) -> Dict[str, Any]:
    data = _load_strategy_latest()
    matches = [r for r in data.get("top5", []) if r.get("symbol") == ticker]

    return {
        "generated_at": _now(),
        "ticker": ticker,
        "found": bool(matches),
        "analysis": matches[0] if matches else {},
    }
