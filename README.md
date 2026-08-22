# Scope — Auto-EDA

Upload any tabular dataset (CSV, TSV, Excel, JSON, Parquet) and get an instant
exploratory data analysis: overview stats, per-column breakdowns, distributions,
correlations, and outliers — no configuration needed.

## Project structure

```
auto-eda/
├── backend/
│   ├── main.py           # FastAPI app — /api/analyze endpoint, serves frontend
│   ├── analysis.py        # Core EDA engine (pandas-based)
│   └── requirements.txt
├── frontend/
│   ├── index.html          # Single-page UI (vanilla JS)
│   └── plotly.min.js       # Charting library, bundled locally (no CDN needed)
└── README.md
```

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd backend
python main.py
```

Or, if you'd rather use the uvicorn CLI directly (same result):

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. The backend serves both the
API and the frontend from the same origin, so there's no separate frontend
server or CORS setup needed for local use.

## How it works

1. You drop a file onto the upload zone (or click to browse).
2. The frontend sends it to `POST /api/analyze` as multipart form data.
3. `analysis.py` loads it into a pandas DataFrame (auto-detecting delimiter
   for CSV/TSV, sheet for Excel, etc.), then computes:
   - Dataset overview: row/column counts, memory footprint, missing %, duplicates
   - Per-column stats: mean/median/std/quartiles for numeric columns, top values
     for categorical columns, length stats for free text, date ranges for datetimes
   - Histograms for every numeric column
   - IQR-based outlier detection per numeric column
   - A full correlation matrix plus the 8 strongest pairwise correlations
   - A 10-row data preview
4. Everything comes back as one JSON payload, which the frontend renders as
   column cards, bar charts, and a correlation heatmap (via Plotly.js).

## Extending it

- **Bigger files**: right now analysis runs synchronously in the request. For
  files much larger than ~50–100MB, move `build_report()` into a background
  job (Celery/RQ + Redis) and poll for the result instead.
- **More formats**: `load_dataframe()` in `analysis.py` is the single place
  to add new format support (e.g. `.feather`, `.h5`).
- **Auth / multi-user**: there's none right now — anyone who can reach the
  server can upload and analyze files. Add auth middleware before deploying
  publicly.
- **Persisting reports**: reports aren't saved anywhere; each upload is
  analyzed fresh. Add a database layer if you want history or shareable links.

## Notes on limits

- Max upload size defaults to 100MB (`MAX_FILE_SIZE_MB` in `main.py`).
- Correlation is computed on a random sample if the dataset has more than
  200,000 rows, to keep response times reasonable (`max_rows_for_corr` in
  `analysis.py`).
