from __future__ import annotations

from typing import Dict, Any, List
from datetime import datetime, timezone
import random


# ------------------------------------------------------------
# Common utils
# ------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def grade_from_score(score: float) -> str:
    """
    점수 → 등급 변환
    (나중에 자유롭게 조정 가능)
    """
    if score >= 85:
        return "A"
    elif score >= 80:
        return "A-"
    elif score >= 75:
        return "B+"
    elif score >= 70:
        return "B"
    elif score >= 65:
        return "B-"
    else:
        return "C"


def prob_from_score(score: float) -> Dict[str, float]:
    """
    점수 → (상승/횡보/하락) 확률 변환
    """
    up = min(max((score - 50) / 50, 0.05), 0.85)
    down = min(max((85 - score) / 50, 0.05), 0.80)
    flat = max(1.0 - up - down, 0.05)

    total = up + flat + down
    return {
        "up": round(up / total, 2),
        "flat": round(flat / total, 2),
        "down": round(down / total, 2),
    }


# ------------------------------------------------------------
# Dummy universe (지금은 테스트용)
# 나중에: 실제 유니버스/엔진 결과로 교체
# ------------------------------------------------------------
DUMMY_UNIVERSE = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("035420", "NAVER"),
    ("035720", "카카오"),
    ("051910", "LG화학"),
    ("068270", "셀트리온"),
    ("096770", "SK이노베이션"),
    ("105560", "KB금융"),
    ("055550", "신한지주"),
    ("012330", "현대모비스"),
]


# ------------------------------------------------------------
# Public API (app.py에서 호출)
# ------------------------------------------------------------
def recommend_top20() -> Dict[str, Any]:
    """
    안정적인 수익 기대 Top20
    """
    items: List[Dict[str, Any]] = []

    for ticker, name in DUMMY_UNIVERSE:
        score = random.uniform(70, 88)

        items.append(
            {
                "ticker": ticker,
                "name": name,
                "score": round(score, 2),
                "grade": grade_from_score(score),
                "prob": prob_from_score(score),
                "score_bin": f"{int(score//5)*5}-{int(score//5)*5+5}",
            }
        )

    # 점수 기준 정렬
    items.sort(key=lambda x: x["score"], reverse=True)

    return {
        "asof": now_iso(),
        "mode": "eod",
        "horizon_days": 20,
        "count": min(20, len(items)),
        "items": items[:20],
    }


def recommend_highrisk5() -> Dict[str, Any]:
    """
    고위험 · 고수익 후보 Top5
    """
    items: List[Dict[str, Any]] = []

    for ticker, name in DUMMY_UNIVERSE:
        score = random.uniform(60, 82)

        items.append(
            {
                "ticker": ticker,
                "name": name,
                "score": round(score, 2),
                "grade": grade_from_score(score),
                "prob": prob_from_score(score),
                "risk_note": "변동성 높음",
            }
        )

    items.sort(key=lambda x: x["score"], reverse=True)

    return {
        "asof": now_iso(),
        "mode": "eod",
        "horizon_days": 20,
        "risk_tag": "high",
        "count": 5,
        "items": items[:5],
    }


def analyze_ticker(ticker: str) -> Dict[str, Any]:
    """
    단일 종목 상세 분석
    """
    score = random.uniform(65, 90)

    return {
        "ticker": ticker,
        "name": None,
        "asof": now_iso(),
        "horizon_days": 20,
        "score": round(score, 2),
        "grade": grade_from_score(score),
        "prob": prob_from_score(score),
        "n_samples": random.randint(300, 3000),
        "levels": {
            "entry": None,
            "stop": None,
            "tp1": None,
            "tp2": None,
        },
        "scenarios": [
            {"type": "up", "desc": "상승 시 분할매도 고려"},
            {"type": "flat", "desc": "횡보 시 관망"},
            {"type": "down", "desc": "이탈 시 손절 고려"},
        ],
        "reasons_top3": [
            {"factor": "trend", "note": "중기 추세 양호"},
            {"factor": "volume", "note": "거래량 필터 통과"},
            {"factor": "momentum", "note": "모멘텀 개선"},
        ],
    }
