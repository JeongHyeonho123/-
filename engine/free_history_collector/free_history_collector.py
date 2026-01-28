# 1.free_history_collector.py
# -*- coding: utf-8 -*-
"""
무료 전종목 히스토리 Collector (Phase 1) - 기업식 2차(MINIMAL, KR 안정화 완성본)
- 국내(KRX 전종목): FinanceDataReader (FDR)
- 미국(전종목): nasdaqtrader 심볼리스트 + yfinance

저장:
  data/history_free/KR/{code}.csv
  data/history_free/US/{ticker}.csv
상태 저장(재시작/재개):
  data/history_free/_state.json

[2차 기업식 최소추가]
- 설정파일 기반(없으면 자동 생성)
- 전역 레이트리밋(RPS) + 요청 캐시(심볼리스트)
- 백오프/재시도 정책 통일
- 상태 저장 강화(done/fail + counters)
- 로그 파일 + 콘솔

[✅ KR 심볼 로딩 안정화(최소수정 완성)]
- FDR StockListing("KRX")가 JSONDecodeError 등으로 막히는 케이스 대응
  1) FDR(KRX) 재시도(백오프)
  2) FDR 분산 소스(KOSPI/KOSDAQ/KONEX) 1회씩 시도 후 병합
  3) PYKRX fallback (설치되어 있으면)
  4) 캐시(data/history_free/_krx_symbols_cache.csv) fallback
"""

import os
import sys

def BASE_DIR():
    """
    ✅ Render/GitHub 서버 기준으로 data/ 경로가 항상 '레포 루트'에 생기도록 통일
    - 현재 파일 위치: engine/free_history_collector/1.free_history_collector.py
    - 레포 루트: 위로 2단계
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def P(*paths):
    return os.path.join(BASE_DIR(), *paths)

def ABS(path: str) -> str:
    if not path:
        return path
    return path if os.path.isabs(path) else P(path)

import json
import time
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import requests
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf


################################################################################
# 기업식 유틸(레이트리밋/백오프/캐시)
################################################################################
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class Logger:
    def __init__(self, path: str):
        self.path = ABS(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.lock = threading.Lock()

    def log(self, msg: str):
        line = f"[{now_ts()}] {msg}"
        print(line)
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

class TokenBucket:
    """초당 N회 정도로 제한(스레드 안전)"""
    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        self.rate = max(0.1, float(rate_per_sec))
        self.capacity = float(capacity) if capacity is not None else self.rate
        self.tokens = self.capacity
        self.updated = time.time()
        self.lock = threading.Lock()

    def acquire(self, cost: float = 1.0):
        cost = float(cost)
        while True:
            with self.lock:
                now = time.time()
                dt = max(0.0, now - self.updated)
                self.updated = now
                self.tokens = min(self.capacity, self.tokens + dt * self.rate)
                if self.tokens >= cost:
                    self.tokens -= cost
                    return
                need = cost - self.tokens
                wait = need / self.rate if self.rate > 0 else 0.2
            time.sleep(min(0.5, max(0.01, wait)))

class BackoffPolicy:
    def __init__(self, retries=3, base_sec=1.0, max_sec=15.0):
        self.retries = max(0, int(retries))
        self.base = max(0.05, float(base_sec))
        self.max = max(self.base, float(max_sec))

    def sleep_time(self, attempt: int):
        t = min(self.max, self.base * (2 ** max(0, attempt)))
        # 아주 가벼운 지터
        j = 0.85 + 0.3 * (time.time() % 1.0)
        return t * j

class SimpleTTLCache:
    def __init__(self, ttl_sec: float = 300.0):
        self.ttl = max(0.0, float(ttl_sec))
        self.data = {}
        self.lock = threading.Lock()

    def get(self, key):
        if self.ttl <= 0:
            return None
        with self.lock:
            v = self.data.get(key)
            if not v:
                return None
            exp, val = v
            if time.time() > exp:
                self.data.pop(key, None)
                return None
            return val

    def set(self, key, val):
        if self.ttl <= 0:
            return
        with self.lock:
            self.data[key] = (time.time() + self.ttl, val)


################################################################################
# 기본 설정
################################################################################
DEFAULT_CONFIG = {
    "output_root": "data/history_free",
    "markets": {
        "KR": {"enabled": True, "years": 5, "max_symbols": 0, "sleep_sec_per_symbol": 0.2},
        "US": {"enabled": True, "years": 10, "max_symbols": 0, "sleep_sec_per_symbol": 0.4},
    },
    "workers": 4,
    "retry": {"max": 3, "backoff_sec": 2.0, "backoff_max_sec": 15.0},
    "rate_limit": {
        "enabled": True,
        "global_rps": 8.0,              # ✅ 전체 네트워크 호출 상한
        "nasdaq_list_cache_ttl": 3600,  # ✅ 심볼리스트 1시간 캐시
    },
    "log_file": "data/history_free/_collector.log",
    "state_file": "data/history_free/_state.json",
    "resume": True,
    "config_file": "free_history_config.json",

    # ✅ KR 심볼 캐시 파일
    "krx_cache_file": "data/history_free/_krx_symbols_cache.csv",
}

def read_json(path: str, default=None):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def merge_dict(base: dict, loaded: dict):
    for k, v in (loaded or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge_dict(base[k], v)
        else:
            base[k] = v

def load_config() -> dict:
    cfg_path = ABS(DEFAULT_CONFIG["config_file"])
    if not os.path.exists(cfg_path):
        d = os.path.dirname(cfg_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)

    loaded = read_json(cfg_path, default={}) or {}
    merged = json.loads(json.dumps(DEFAULT_CONFIG))  # deep-ish copy
    merge_dict(merged, loaded)

    # ✅ dist 기준 절대경로 보정
    for k in ("output_root", "log_file", "state_file", "config_file", "krx_cache_file"):
        if isinstance(merged.get(k), str):
            merged[k] = ABS(merged[k])

    return merged


################################################################################
# 상태 저장
################################################################################
class StateStore:
    def __init__(self, path: str):
        self.path = ABS(path)
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {
                "done": {"KR": [], "US": []},
                "fail": {"KR": {}, "US": {}},
                "stats": {"ok": 0, "fail": 0, "skip": 0, "updated": 0},
                "updated_at": now_ts()
            }
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                raise ValueError("state invalid")
            d.setdefault("done", {"KR": [], "US": []})
            d.setdefault("fail", {"KR": {}, "US": {}})
            d.setdefault("stats", {"ok": 0, "fail": 0, "skip": 0, "updated": 0})
            d.setdefault("updated_at", now_ts())
            return d
        except Exception:
            return {
                "done": {"KR": [], "US": []},
                "fail": {"KR": {}, "US": {}},
                "stats": {"ok": 0, "fail": 0, "skip": 0, "updated": 0},
                "updated_at": now_ts()
            }

    def save(self):
        with self.lock:
            self.state["updated_at"] = now_ts()
            self.state["stats"]["updated"] = int(self.state["stats"].get("updated", 0)) + 1
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)

    def is_done(self, market: str, symbol: str) -> bool:
        return symbol in set(self.state.get("done", {}).get(market, []))

    def mark_done(self, market: str, symbol: str):
        with self.lock:
            done_list = self.state.setdefault("done", {}).setdefault(market, [])
            if symbol not in done_list:
                done_list.append(symbol)
            self.state["stats"]["ok"] = int(self.state["stats"].get("ok", 0)) + 1

    def mark_skip(self):
        with self.lock:
            self.state["stats"]["skip"] = int(self.state["stats"].get("skip", 0)) + 1

    def mark_fail(self, market: str, symbol: str, reason: str):
        with self.lock:
            fail_map = self.state.setdefault("fail", {}).setdefault(market, {})
            fail_map[symbol] = {"reason": reason, "ts": now_ts()}
            self.state["stats"]["fail"] = int(self.state["stats"].get("fail", 0)) + 1

    def clear_fail(self, market: str, symbol: str):
        with self.lock:
            fail_map = self.state.setdefault("fail", {}).setdefault(market, {})
            fail_map.pop(symbol, None)


################################################################################
# 심볼 로더 (✅ KR 안정화 완성: FDR -> PYKRX -> CACHE)
################################################################################
def _krx_codes_from_df(df: pd.DataFrame) -> List[str]:
    if df is None or getattr(df, "empty", True):
        return []
    for col in ["Code", "code", "종목코드"]:
        if col in df.columns:
            codes = df[col].astype(str).str.strip().str.zfill(6).tolist()
            codes = [c for c in codes if c.isdigit() and len(c) == 6]
            return sorted(list(set(codes)))
    return []

def _read_krx_cache(cache_path: str) -> List[str]:
    try:
        if not cache_path or not os.path.exists(cache_path):
            return []
        df = pd.read_csv(cache_path)
        return _krx_codes_from_df(df)
    except Exception:
        return []

def _write_krx_cache(cache_path: str, codes: List[str]):
    try:
        if not cache_path:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pd.DataFrame({"Code": codes}).to_csv(cache_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass

def _try_pykrx(logger: Logger) -> List[str]:
    """
    pykrx 기반 KOSPI/KOSDAQ/KONEX 상장목록
    - FDR이 막혀도 pykrx가 되는 경우가 꽤 있음
    """
    try:
        from pykrx import stock
    except Exception as e:
        logger.log(f"[KR] pykrx import fail: {repr(e)}")
        return []

    try:
        kospi = stock.get_market_ticker_list(market="KOSPI") or []
        kosdaq = stock.get_market_ticker_list(market="KOSDAQ") or []
        konex = stock.get_market_ticker_list(market="KONEX") or []
        merged = sorted(list(set([str(x).zfill(6) for x in (kospi + kosdaq + konex)])))
        merged = [c for c in merged if c.isdigit() and len(c) == 6]
        logger.log(
            f"[KR] symbol source=PYKRX merged={len(merged)} "
            f"(KOSPI={len(kospi)} KOSDAQ={len(kosdaq)} KONEX={len(konex)})"
        )
        return merged
    except Exception as e:
        logger.log(f"[KR] pykrx ticker load fail: {repr(e)}")
        return []

def load_krx_symbols(max_symbols: int, logger: Logger, backoff: BackoffPolicy, cache_path: str) -> List[str]:
    def _try_fdr(market_name: str) -> List[str]:
        df = fdr.StockListing(market_name)
        return _krx_codes_from_df(df)

    last_err = None

    # 1) FDR(KRX) 재시도
    for attempt in range(backoff.retries + 1):
        try:
            codes = _try_fdr("KRX")
            if codes:
                logger.log(f"[KR] symbol source=FDR(KRX) ok codes={len(codes)}")
                _write_krx_cache(cache_path, codes)
                return codes[:max_symbols] if (max_symbols and max_symbols > 0) else codes
            last_err = "empty codes from FDR(KRX)"
            logger.log(f"[KR] symbol source=FDR(KRX) empty (attempt {attempt+1})")
        except Exception as e:
            last_err = repr(e)
            logger.log(f"[KR] FDR(KRX) retry {attempt+1}/{backoff.retries+1} err={last_err}")
        time.sleep(backoff.sleep_time(attempt))

    # 2) FDR 분산(KOSPI/KOSDAQ/KONEX) 1회씩
    merged = set()
    ok_any = False
    for mkt in ["KOSPI", "KOSDAQ", "KONEX"]:
        try:
            codes = _try_fdr(mkt)
            if codes:
                ok_any = True
                merged.update(codes)
                logger.log(f"[KR] symbol source=FDR({mkt}) ok codes={len(codes)}")
            else:
                logger.log(f"[KR] symbol source=FDR({mkt}) empty")
        except Exception as e:
            logger.log(f"[KR] symbol source=FDR({mkt}) fail err={repr(e)}")

    merged_list = sorted(list(merged))
    if ok_any and merged_list:
        logger.log(f"[KR] symbol source=FDR(KOSPI+KOSDAQ+KONEX) merged={len(merged_list)}")
        _write_krx_cache(cache_path, merged_list)
        return merged_list[:max_symbols] if (max_symbols and max_symbols > 0) else merged_list

    # 3) PYKRX fallback
    pykrx_codes = _try_pykrx(logger)
    if pykrx_codes:
        _write_krx_cache(cache_path, pykrx_codes)
        return pykrx_codes[:max_symbols] if (max_symbols and max_symbols > 0) else pykrx_codes

    # 4) CACHE fallback
    cached = _read_krx_cache(cache_path)
    if cached:
        logger.log(f"[KR] symbol source=CACHE ok codes={len(cached)} path={cache_path}")
        logger.log(f"[KR] NOTE: live listing failed. last_err={last_err}")
        return cached[:max_symbols] if (max_symbols and max_symbols > 0) else cached

    raise RuntimeError(f"KR symbol load failed. last_err={last_err} (cache_empty path={cache_path})")

def load_us_symbols_from_nasdaqtrader(
    max_symbols: int = 0,
    session: Optional[requests.Session] = None,
    bucket: Optional[TokenBucket] = None,
    cache: Optional[SimpleTTLCache] = None,
    logger: Optional[Logger] = None
) -> List[str]:
    """
    나스닥 트레이더(무료 공개)에서 상장 심볼 목록 수집
    - nasdaqlisted.txt: NASDAQ 상장
    - otherlisted.txt: NYSE/AMEX 등 기타
    """
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    session = session or requests.Session()

    key = "nasdaqtrader_symdir_v1"
    if cache:
        cached = cache.get(key)
        if cached:
            if logger:
                logger.log(f"[US] nasdaqtrader symbol list cache hit: {len(cached)}")
            return cached[:max_symbols] if (max_symbols and max_symbols > 0) else cached

    symbols = []
    for url in urls:
        if bucket:
            bucket.acquire(1.0)
        r = session.get(url, timeout=30)
        r.raise_for_status()
        text = r.text.splitlines()
        rows = [line for line in text if "|" in line and not line.startswith("File Creation Time")]
        if not rows:
            continue
        header = rows[0].split("|")
        sym_idx = header.index("Symbol") if "Symbol" in header else 0
        for line in rows[1:]:
            parts = line.split("|")
            if len(parts) <= sym_idx:
                continue
            sym = parts[sym_idx].strip()
            if not sym or "^" in sym or "/" in sym:
                continue
            symbols.append(sym)

    symbols = sorted(list(set(symbols)))
    if cache:
        cache.set(key, symbols)
    return symbols[:max_symbols] if (max_symbols and max_symbols > 0) else symbols


################################################################################
# 저장 유틸
################################################################################
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_df_as_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")


################################################################################
# 수집 함수
################################################################################
def fetch_kr_daily(code: str, years: int) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=365 * years + 30)
    df = fdr.DataReader(code, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        raise RuntimeError("빈 데이터")
    df = df.reset_index()
    if "Date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df

def fetch_us_daily(ticker: str, years: int, bucket: Optional[TokenBucket] = None) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=365 * years + 30)
    if bucket:
        bucket.acquire(1.0)
    df = yf.download(
        tickers=ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError("빈 데이터")
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df


################################################################################
# 워커
################################################################################
@dataclass
class Job:
    market: str
    symbol: str
    years: int
    sleep_sec: float

def worker_loop(
    job_q: "queue.Queue[Job]",
    cfg: dict,
    logger: Logger,
    state: StateStore,
    bucket: Optional[TokenBucket],
    backoff: BackoffPolicy
):
    out_root = cfg["output_root"]

    while True:
        try:
            job = job_q.get(timeout=1.0)
        except queue.Empty:
            return

        market = job.market
        sym = job.symbol

        try:
            if cfg.get("resume", True) and state.is_done(market, sym):
                logger.log(f"[SKIP] {market} {sym} (already done)")
                state.mark_skip()
                job_q.task_done()
                continue

            ok = False
            last_err = None

            for attempt in range(backoff.retries + 1):
                try:
                    if market == "KR":
                        df = fetch_kr_daily(sym, job.years)
                        out_path = os.path.join(out_root, "KR", f"{sym}.csv")
                    else:
                        df = fetch_us_daily(sym, job.years, bucket=bucket)
                        out_path = os.path.join(out_root, "US", f"{sym}.csv")

                    save_df_as_csv(df, out_path)
                    state.clear_fail(market, sym)
                    state.mark_done(market, sym)
                    state.save()

                    logger.log(f"[OK] {market} {sym} rows={len(df)} -> {out_path}")
                    ok = True
                    break

                except Exception as e:
                    last_err = str(e)
                    logger.log(f"[RETRY {attempt+1}/{backoff.retries+1}] {market} {sym} err={e}")
                    time.sleep(backoff.sleep_time(attempt))

            if not ok:
                state.mark_fail(market, sym, last_err or "unknown")
                state.save()
                logger.log(f"[FAIL] {market} {sym} reason={last_err}")

            time.sleep(float(job.sleep_sec))

        finally:
            job_q.task_done()


################################################################################
# 메인
################################################################################
def main():
    cfg = load_config()

    logger = Logger(cfg["log_file"])
    state = StateStore(cfg["state_file"])

    rl = cfg.get("rate_limit", {}) or {}
    bucket = None
    if bool(rl.get("enabled", True)):
        rps = float(rl.get("global_rps", 8.0))
        bucket = TokenBucket(rate_per_sec=rps, capacity=rps)

    back = cfg.get("retry", {}) or {}
    backoff = BackoffPolicy(
        retries=int(back.get("max", 3)),
        base_sec=float(back.get("backoff_sec", 2.0)),
        max_sec=float(back.get("backoff_max_sec", 15.0)),
    )

    cache = SimpleTTLCache(ttl_sec=float(rl.get("nasdaq_list_cache_ttl", 3600)))

    logger.log("무료 히스토리 Collector 시작")
    logger.log(f"BASE_DIR={BASE_DIR()}")
    logger.log(f"output_root={cfg['output_root']} workers={cfg['workers']} resume={cfg.get('resume', True)}")
    logger.log(f"rate_limit_enabled={bool(rl.get('enabled', True))} global_rps={rl.get('global_rps', 8.0)}")
    logger.log(f"retry={backoff.retries} backoff_base={backoff.base} backoff_max={backoff.max}")
    logger.log(f"krx_cache_file={cfg.get('krx_cache_file')}")

    job_q: "queue.Queue[Job]" = queue.Queue()

    # KR
    if cfg["markets"]["KR"]["enabled"]:
        try:
            kr_syms = load_krx_symbols(
                int(cfg["markets"]["KR"]["max_symbols"]),
                logger=logger,
                backoff=backoff,
                cache_path=str(cfg.get("krx_cache_file", "")),
            )
            logger.log(f"[KR] symbols={len(kr_syms)} years={cfg['markets']['KR']['years']}")
            for s in kr_syms:
                job_q.put(Job("KR", s, int(cfg["markets"]["KR"]["years"]),
                              float(cfg["markets"]["KR"]["sleep_sec_per_symbol"])))
        except Exception as e:
            logger.log(f"[KR] symbol load fail: {e}")

    # US
    if cfg["markets"]["US"]["enabled"]:
        try:
            session = requests.Session()
            us_syms = load_us_symbols_from_nasdaqtrader(
                int(cfg["markets"]["US"]["max_symbols"]),
                session=session, bucket=bucket, cache=cache, logger=logger
            )
            logger.log(f"[US] symbols={len(us_syms)} years={cfg['markets']['US']['years']}")
            for s in us_syms:
                job_q.put(Job("US", s, int(cfg["markets"]["US"]["years"]),
                              float(cfg["markets"]["US"]["sleep_sec_per_symbol"])))
        except Exception as e:
            logger.log(f"[US] symbol load fail: {e}")

    if job_q.qsize() == 0:
        logger.log("수집할 작업이 없습니다. 설정을 확인하세요.")
        return

    workers = max(1, int(cfg["workers"]))
    threads = []
    for _ in range(workers):
        t = threading.Thread(target=worker_loop, args=(job_q, cfg, logger, state, bucket, backoff), daemon=True)
        t.start()
        threads.append(t)

    last = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(1.0)
        if time.time() - last > 10:
            last = time.time()
            st = state.state.get("stats", {})
            logger.log(
                f"[PROGRESS] remaining_jobs={job_q.qsize()} "
                f"done_KR={len(state.state['done']['KR'])} done_US={len(state.state['done']['US'])} "
                f"ok={st.get('ok',0)} fail={st.get('fail',0)} skip={st.get('skip',0)}"
            )

    logger.log("무료 히스토리 Collector 종료")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] free_history_collector failed: {e}")
        import traceback
        print(traceback.format_exc())
        raise
