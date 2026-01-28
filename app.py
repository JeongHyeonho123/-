from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/recommend/top20")
def top20():
    return {"result": "top20 dummy"}

@app.get("/recommend/highrisk")
def highrisk():
    return {"result": "highrisk dummy"}

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    return {"ticker": ticker, "analysis": "dummy"}
