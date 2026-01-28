import zipfile

ZIP_NAME = "recommend_engine_src.zip"  # zip 파일명이 다르면 여기만 바꾸기

with zipfile.ZipFile(ZIP_NAME, "r") as zf:
    for name in zf.namelist():
        print(name)
