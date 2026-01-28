from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import os
import re


# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

FOLDERS = {
    "free": ENGINE_DIR / "free_history_collector",
    "norm": ENGINE_DIR / "nomalize_history",
    "research": ENGINE_DIR / "research",
    "signals": ENGINE_DIR / "signals_history_builder",
    "strategy": ENGINE_DIR / "strategy_research",
}

STRATEGY_DIR = DATA_DIR / "strategy"
STRATEGY_LATEST_JSON = STRATEGY_DIR / "strategy_latest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# Module loader by file path
# -----------------------------
def _load_module(py_file: Path):
    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec: {py_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_py_with_main(folder: Path) -> Optional[Path]:
    """
    folder 내에서 main() 함수를 가진 .py 파일을 찾아 반환.
    파일명이 1.xxx.py 처럼 숫자 prefix가 있으면 우선시.
    """
    if not folder.exists():
        return None

    py_files = [p for p in folder.glob("*.py") if p.name != "__init__.py"]
    if not py_files:
        return None

    def score(p: Path) -> int:
        s = 0
        name = p.name.lower()
        # 숫자 prefix 선호
        if re.match(r"^\d+[\._-]", p.name):
            s += 100
        # main 들어가면 가산점
        if "main" in name:
            s += 20
        # 경로 깊이 적을수록 가산(여긴 동일)
        return s

    py_files.sort(key=score, reverse=True)

    for p in py_files:
        try:
            mod = _load_module(p)
            if hasattr(mod, "main") and callable(getattr(mod, "main")):
                return p
        except Exception:
            continue

    return None


def _run_folder_main(folder: Path) -> Dict[str, Any]:
    py = _find_py_with_main(folder)
    if py is None:
        return {"ok": False, "reason": f"main() 가진 .py를 못 찾음: {folder}"}

    try:
        mod = _load_module(py)
        mod.main()
        return {"ok": True, "file": str(py)}
    except Exception as e:
        return {"ok": False, "file": str(py), "reason": str(e)}


# -----------------------------
# Strategy cache loader
# -----------------------------
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_records(path: Path) -> List[Dict[str, Any]]:
    import pandas as pd
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception:
        return []


def _load_latest_strategy_rows() -> Dict[str, Any]:
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    latest = _read_json(STRATEGY_LATEST_JSON)
    if not latest:
        return {"ok": False, "reason": "strategy_latest.json not found", "latest": None, "rows": []}

    csv_name = latest.get("csv")
    if not csv_name:
        return {"ok": False, "reason": "strategy_latest.json has no 'csv' field", "latest": latest, "rows": []}

    csv_path = STRATEGY_DIR / csv_name
    rows = _read_csv_records(csv_path)
    return {"ok": True, "reason": "loaded", "latest": latest, "rows": rows}


# -----------------------------
# Scoring -> Grade/Prob (UI)
# -----------------------------
def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _score_from_row(row: Dict[str, Any]) -> float:
    """
    row에 winrate/cagr/max_dd 등이 있을 때 점수화.
    (현호님 엔진 결과 컬럼이 다르면 여기만 바꾸면 됨)
    """
    winrate = _to_float(row.get("winrate", 0), 0)
    cagr = _to_float(row.get("cagr", 0), 0)
    max_dd = _to_float(row.get("max_dd", 0), 0)
    # score scale: 대략 0~100대
    return (winrate * 100.0) + (cagr * 50.0) - (abs(max_dd) * 50.0)


def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 80:
        return "A-"
    if score >= 75:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 65:
        return "B-"
    return "C"


def _prob(score: float) -> Dict[str, float]:
    """
    점수 -> 상승/횡보/하락 확률(표시용)
    """
    up = min(max((score - 50.0) / 50.0, 0.05), 0.85)
    down = min(max((85.0 - score) / 50.0, 0.05), 0.80)
    flat = max(1.0 - up - down, 0.05)
    total = up + flat + down
    return {
        "up": round(up / total, 2),
        "flat": round(flat / total, 2),
        "down": round(down / total, 2),
    }


def _bin(score: float, step: int = 5) -> str:
    lo = int(score // step) * step
    hi = lo + step
    return f"{lo}-{hi}"


# -----------------------------
# Pipeline runner (on-demand)
# -----------------------------
def run_pipeline() -> Dict[str, Any]:
    """
    현호님 5단계 파이프라인을 폴더 기준으로 main() 실행.
    순서:
      1) free_history_collector
      2) nomalize_history
      3) signals_history_builder
      4) research
      5) strategy_research
    """
    results = {}
    order = ["free", "norm", "signals", "research", "strategy"]
    for k in order:
        results[k] = _run_folder_main(FOLDERS[k])
        if not results[k]["ok"]:
            return {"ok": False, "asof": now_iso(), "results": results}
    return {"ok": True, "asof": now_iso(), "results": results}


# -----------------------------
# Public APIs (called by app.py)
# -----------------------------
def recommend_top20(force_run: bool = False) -> Dict[str, Any]:
    """
    Top20 추천:
      - 캐시(strategy_latest.json + csv) 있으면 읽어서 Top20 반환
      - 없으면 force_run=True일 때만 파이프라인 실행 후 재시도
    """
    loaded = _load_latest_strategy_rows()

    pipeline = None
    if (not loaded["ok"]) and force_run:
        pipeline = run_pipeline()
        loaded = _load_latest_strategy_rows()

    if not loaded["ok"]:
        return {
            "asof": now_iso(),
            "mode": "eod",
            "horizon_days": 20,
            "engine": "local-folders",
            "pipeline": pipeline,
            "items": [],
            "note": "strategy 캐시가 없습니다. 먼저 파이프라인을 1회 실행해 캐시를 생성하세요.",
        }

    rows: List[Dict[str, Any]] = loaded["rows"]
    items: List[Dict[str, Any]] = []
    for row in rows:
        score = _score_from_row(row)
        items.append(
            {
                "ticker": str(row.get("symbol", row.get("ticker", ""))),
                "name": row.get("name"),
                "market": row.get("market"),
                "strategy": row.get("strategy"),
                "score": round(score, 2),
                "grade": _grade(score),
                "prob": _prob(score),
                "score_bin": _bin(score),
                "stats": {
                    "equity": row.get("equity"),
                    "cagr": row.get("cagr"),
                    "max_dd": row.get("max_dd"),
                    "trades": row.get("trades"),
                    "winrate": row.get("winrate"),
                },
            }
        )

    items.sort(key=lambda x: x["score"], reverse=True)
    return {
        "asof": now_iso(),
        "mode": "eod",
        "horizon_days": 20,
        "engine": "local-folders",
        "generated_at": (loaded["latest"] or {}).get("generated_at"),
        "count": min(20, len(items)),
        "items": items[:20],
    }


def recommend_highrisk5(force_run: bool = False) -> Dict[str, Any]:
    """
    고위험 5개: max_dd 절대값 큰 순(간단 룰)
    """
    top = recommend_top20(force_run=force_run)
    items = top.get("items", [])

    def risk_key(x: Dict[str, Any]) -> float:
        md = x.get("stats", {}).get("max_dd", 0) or 0
        return abs(_to_float(md, 0))

    items_sorted = sorted(items, key=risk_key, reverse=True)
    return {
        "asof": now_iso(),
        "mode": top.get("mode", "eod"),
        "horizon_days": top.get("horizon_days", 20),
        "engine": top.get("engine", "local-folders"),
        "items": items_sorted[:5],
        "note": "High-risk bucket based on larger |max_dd| (adjustable).",
    }


def analyze_ticker(ticker: str, force_run: bool = False) -> Dict[str, Any]:
    """
    단일 종목 분석:
      - strategy 캐시에서 ticker를 찾아 점수/등급/확률 반환
    """
    ticker = ticker.strip()

    loaded = _load_latest_strategy_rows()
    pipeline = None
    if (not loaded["ok"]) and force_run:
        pipeline = run_pipeline()
        loaded = _load_latest_strategy_rows()

    if not loaded["ok"]:
        return {
            "ticker": ticker,
            "asof": now_iso(),
            "engine": "local-folders",
            "pipeline": pipeline,
            "note": "strategy 캐시가 없습니다. 먼저 파이프라인을 1회 실행해 캐시를 생성하세요.",
        }

    rows: List[Dict[str, Any]] = loaded["rows"]
    target = None
    for r in rows:
        sym = str(r.get("symbol", r.get("ticker", ""))).strip()
        if sym == ticker:
            target = r
            break

    if not target:
        return {
            "ticker": ticker,
            "asof": now_iso(),
            "engine": "local-folders",
            "note": "ticker가 최신 strategy 캐시에 없습니다(유니버스/마켓 불일치 가능).",
        }

    score = _score_from_row(target)
    return {
        "ticker": ticker,
        "name": target.get("name"),
        "asof": now_iso(),
        "horizon_days": 20,
        "engine": "local-folders",
        "score": round(score, 2),
        "grade": _grade(score),
        "prob": _prob(score),
        "score_bin": _bin(score),
        "n_samples": int(_to_float(target.get("trades", 0), 0)),
        "raw": {
            "market": target.get("market"),
            "strategy": target.get("strategy"),
            "equity": target.get("equity"),
            "cagr": target.get("cagr"),
            "max_dd": target.get("max_dd"),
            "trades": target.get("trades"),
            "winrate": target.get("winrate"),
        },
        "levels": {
            "entry": None,
            "stop": None,
            "tp1": None,
            "tp2": None,
        },
        "reasons_top3": [
            {"factor": "strategy", "note": f"strategy={target.get('strategy')}"},
            {"factor": "winrate", "note": f"winrate={target.get('winrate')}"},
            {"factor": "cagr/dd", "note": f"cagr={target.get('cagr')} / max_dd={target.get('max_dd')}"},
        ],
    }
