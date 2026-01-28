from fastapi import FastAPI
from engine.research import hello_engine

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok", "engine": hello_engine()}

@app.get("/recommend/top20")
def top20():
    return {"result": "top20 dummy", "engine": hello_engine()}

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    return {"ticker": ticker, "analysis": "dummy", "engine": hello_engine()}
