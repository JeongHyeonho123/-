import re
import sys
import zipfile
from pathlib import Path

# ✅ 여기 zip 파일명만 맞추면 됨
ZIP_NAME = "recommend_engine_src.zip"

# (목표 파일) zip 안에서 이 이름이 들어간 .py를 자동으로 찾아서 매핑
TARGETS = {
    "free_history_collector": "engine/legacy/free_history_collector.py",
    "normalize_history": "engine/legacy/normalize_history.py",
    "nomalize_history": "engine/legacy/normalize_history.py",  # 오타 폴더도 대응
    "research": "engine/legacy/research.py",
    "strategy_research": "engine/legacy/strategy_research.py",
    "signals_history_builder": "engine/legacy/signals_history_builder.py",
}

BASE_DIR_REPLACEMENT = """def BASE_DIR():
    import os
    # engine/legacy/xxx.py 기준 -> 레포 루트는 2단계 위
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""

def patch_base_dir(src: str) -> str:
    # BASE_DIR 함수가 있으면 교체, 없으면 맨 위에 삽입
    pattern = r"def BASE_DIR\(\):\s*.*?(?=\n\s*\ndef |\n\s*\n)"
    m = re.search(pattern, src, flags=re.DOTALL)
    if not m:
        return BASE_DIR_REPLACEMENT + "\n" + src
    start, end = m.span()
    return src[:start] + BASE_DIR_REPLACEMENT + "\n" + src[end:]


def ensure_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def find_best_match(namelist, keyword: str) -> str | None:
    """
    zip 내부 경로 중에서 keyword가 포함된 .py 파일을 찾아서
    가장 그럴듯한 1개를 반환.
    """
    candidates = [n for n in namelist if n.lower().endswith(".py") and keyword.lower() in n.lower()]

    if not candidates:
        return None

    # 우선순위: 숫자 prefix가 붙은 파일(예: 1.xxx.py, 2.xxx.py)을 선호
    def score(name: str) -> int:
        base = name.split("/")[-1]
        s = 0
        if re.match(r"^\d+[\._-]", base):
            s += 100
        # 폴더명 정확히 포함되면 가산점
        if f"{keyword.lower()}/" in name.lower():
            s += 50
        # 너무 깊은 경로는 감점(대충)
        s -= name.count("/")
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def main():
    repo_root = Path(__file__).resolve().parent
    zip_path = repo_root / ZIP_NAME

    if not zip_path.exists():
        print(f"[ERROR] zip 파일이 없습니다: {zip_path}")
        print("zip 파일명을 확인해서 ZIP_NAME을 맞춰주세요.")
        sys.exit(1)

    # engine/legacy 패키지 뼈대 생성
    ensure_file(repo_root / "engine" / "__init__.py")
    ensure_file(repo_root / "engine" / "legacy" / "__init__.py")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # 디버그: zip 내부에 뭐가 있는지 일부 보여줌(너무 길면 생략)
        py_files = [n for n in names if n.lower().endswith(".py")]
        if not py_files:
            print("[ERROR] zip 안에 .py 파일이 하나도 없습니다.")
            sys.exit(1)

        found = {}
        for key, dst in TARGETS.items():
            match = find_best_match(names, key)
            if match:
                found[key] = (match, dst)

        # 키워드 중복(예: normalize_history / nomalize_history) 정리
        # 둘 다 찾았으면 더 점수 높은 쪽만 사용
        if "normalize_history" in found and "nomalize_history" in found:
            # normalize_history 우선 사용, nomalize_history는 제거
            found.pop("nomalize_history", None)

        required_keys = ["free_history_collector", "normalize_history", "research", "strategy_research", "signals_history_builder"]
        missing = [k for k in required_keys if k not in found]

        if missing:
            print("[ERROR] 자동 탐색으로도 필요한 파일을 못 찾았습니다:")
            for m in missing:
                print(" -", m)
            print("\n[HINT] zip 내부 파일 목록을 보고 싶으면 아래 실행:")
            print("python -c \"import zipfile; z=zipfile.ZipFile('"+ZIP_NAME+"'); print('\\n'.join(z.namelist()))\"")
            sys.exit(1)

        # 파일 생성
        for key in required_keys:
            src_name, dst_name = found[key]
            raw = zf.read(src_name).decode("utf-8", errors="replace")
            patched = patch_base_dir(raw)

            dst_path = repo_root / dst_name
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            dst_path.write_text(patched, encoding="utf-8")
            print(f"[OK] {key}: {src_name} -> {dst_name}")

    print("\n[DONE] legacy engine files prepared.")
    print("Next: GitHub commit & push -> Render Deploy latest commit.")


if __name__ == "__main__":
    main()
