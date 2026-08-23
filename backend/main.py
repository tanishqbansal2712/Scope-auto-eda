"""
main.py
FastAPI app for the Auto-EDA website.

Run either way:
  python main.py
  uvicorn main:app --reload --port 8000
Then open http://localhost:8000 in your browser.
"""

import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from analysis import build_report, load_dataframe

# 25MB, not the 100MB this used to say. Render's free tier caps a service
# at 512MB of RAM total (shared with Python/FastAPI/pandas itself), and
# pandas typically uses several times a CSV's on-disk size once it's
# loaded into a DataFrame plus the analysis this app runs on top of it.
# A 60-80MB file was enough to exceed that and get the whole process
# killed mid-request — which the browser sees as an empty response, not
# a clean error message. This cap keeps uploads within what the free tier
# can reliably handle. If you're running this locally, or have upgraded
# to a paid Render plan with more RAM, raise this back up.
MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}

app = FastAPI(title="Auto-EDA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f} MB; the limit is {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        df = load_dataframe(file.filename, raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=422, detail="The uploaded file produced an empty dataset.")

    try:
        report = build_report(df)
    except Exception as exc:
        # Print the full traceback to the terminal running this server —
        # the HTTP response only shows str(exc), which isn't enough to
        # locate the failing line. Check the terminal for the real cause.
        print("\n" + "=" * 70)
        print("ANALYSIS FAILED — full traceback:")
        traceback.print_exc()
        print("=" * 70 + "\n")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    report["filename"] = file.filename
    return report


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the frontend (index.html + static assets) from the same origin
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
