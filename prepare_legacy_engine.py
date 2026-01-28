import re
import sys
import zipfile
from pathlib import Path

ZIP_NAME = "recommend_engine_src.zip"

# zip 안 원본 파일 → repo에 만들 파일 경로
MAPPING = {
    "free_history_collector/1.free_history_collector.py": "engine/legacy/free_history_collector.py",
    "nomalize_history/2.normalize_history.py": "engine/legacy/normalize_history.py",
    "research/3.research.py": "engine/legacy/research.py",
    "strategy_research/4.strategy_research.py": "engine/legacy/strategy_research.py",
    "signals_history_builder/12.signals_history_builder.py": "engine/legacy/signals_history_builder.py",
}

BASE_DIR_REPLACEMENT = """def BASE_DIR():
    import os
    # engine/legacy/xxx.py 기준 -> 레포 루트는 2단계 위
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""

def patch_base_dir(src: str) -> str:
    """
    기존 BASE_DIR() 함수 전체를 교체한다.
    - 패턴: def BASE_DIR(): 부터 다음 빈 줄까지(또는 다음 def 전까지) 넉넉히 잡아 교체
    """
    # 가장 흔한 형태: def BASE_DIR(): ... return ... (뒤에 빈 줄)
    pattern = r"def BASE_DIR\(\):\s*.*?(?=\n\s*\ndef |\n\s*\ndef P\(|\n\s*def P\(|\n\s*\n)"
    m = re.search(pattern, src, flags=re.DOTALL)
    if not m:
        # BASE_DIR가 없으면 앞쪽에 삽입
        return BASE_DIR_REPLACEMENT + "\n" + src

    start, end = m.span()
    return src[:start] + BASE_DIR_REPLACEMENT + "\n" + src[end:]


def ensure_init(py_path: Path):
    py_path.parent.mkdir(parents=True, exist_ok=True)
    if not py_path.exists():
        py_path.write_text("", encoding="utf-8")


def main():
    repo_root = Path(__file__).resolve().parent
    zip_path = repo_root / ZIP_NAME

    if not zip_path.exists():
        print(f"[ERROR] {ZIP_NAME} 파일이 레포 루트에 없습니다: {zip_path}")
        sys.exit(1)

    # engine/legacy 패키지 생성
    ensure_init(repo_root / "engine" / "__init__.py")
    ensure_init(repo_root / "engine" / "legacy" / "__init__.py")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        missing = [k for k in MAPPING.keys() if k not in names]
        if missing:
            print("[ERROR] zip 안에 필요한 파일이 없습니다:")
            for m in missing:
                print(" -", m)
            sys.exit(1)

        for src_name, dst_name in MAPPING.items():
            raw = zf.read(src_name).decode("utf-8", errors="replace")
            patched = patch_base_dir(raw)

            dst_path = repo_root / dst_name
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            dst_path.write_text(patched, encoding="utf-8")
            print(f"[OK] wrote: {dst_name}")

    print("\n[DONE] legacy engine files prepared.")
    print("Next: commit & push to GitHub, then Render deploy.")


if __name__ == "__main__":
    main()
