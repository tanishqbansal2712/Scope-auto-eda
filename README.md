# Scope — Auto-EDA

Upload any tabular dataset (CSV, TSV, Excel, JSON, Parquet) and get an
instant, thorough exploratory data analysis — no configuration, no code.
One file in, a full dashboard out.

## What it does

**Overview** — row/column counts, memory footprint, missing %, duplicate rows.

**Data quality layer**
- Missing value report with a nullity heatmap and a missingness-correlation
  heatmap (do columns tend to go missing together?)
- Duplicate row detection, with a sample of the actual duplicates
- Per-column outlier detection (IQR-based)
- Type mismatch warnings (a column that's mostly numeric but has a few
  stray non-numeric values)
- Cardinality flags — ID-like columns and high-cardinality traps
- Per-column imputation suggestions based on dtype, missingness, and skew

**Deeper statistics**
- Distribution shape labels (right-skewed, heavy-tailed, etc.) from
  skew/kurtosis
- Shapiro-Wilk normality test per numeric column *(needs optional `scipy`)*
- Chi-square test of independence between categorical column pairs
  *(also needs `scipy`)*
- PCA (2 components) and a KMeans clustering preview with an elbow chart —
  implemented with plain numpy, no scikit-learn dependency
- Time series trend + day-of-week/month-of-year seasonality, if a datetime
  column is present

**ML readiness**
- Auto-detects reasonable target columns (numeric -> regression,
  low-cardinality categorical -> classification)
- Feature importance per target via mutual information (not a trained
  model -- a fast, dependency-light relevance signal)
- Class imbalance detection for classification targets
- Train/test split size preview
- A single 0-100 readiness score with a visible breakdown of what's
  dragging it down

Every column card, chart, and section renders live in the browser --
nothing is written to disk, and nothing about your data leaves your own
backend.

## Project structure

```
auto-eda-app/
├── render.yaml              # Render Blueprint -- one-click backend+frontend deploy
├── README.md
├── backend/
│   ├── main.py                # FastAPI app -- /api/analyze, /api/health, serves frontend
│   ├── analysis.py            # The entire EDA engine (pandas/numpy, optional scipy)
│   └── requirements.txt
└── frontend/
    ├── index.html              # Single-page UI (vanilla JS, no build step)
    └── plotly.min.js           # Charting library, bundled locally (no CDN dependency)
```

## Running it locally

```powershell
cd backend
pip install -r requirements.txt
python main.py
```

Or, equivalently:

```powershell
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** -- the backend serves both the API and the
frontend from the same origin, so there's nothing else to start.

### Optional: scipy

`scipy` enables two specific features -- the Shapiro-Wilk normality test and
the chi-square test -- and nothing else depends on it. It's left out of the
base install because it needs a build toolchain on platforms without a
prebuilt wheel for it.

- **64-bit Python:** `pip install scipy` -- any recent version works.
- **32-bit Windows:** scipy dropped 32-bit Windows wheels after version
  1.8.1, so pin to that exact version to avoid a source build:
  `pip install scipy==1.8.1`

Without it, everything else (data quality, PCA, clustering, time series,
ML readiness) still works fully -- the two affected panels just show a
short note instead.

## Deploying (Render)

The whole app -- frontend and backend -- deploys as a single Render web
service, since `main.py` already serves both from one FastAPI app.

1. Push this project to a GitHub repo, with `render.yaml` at the repo
   root (next to `backend/` and `frontend/`, not inside either).
2. On [render.com](https://render.com), **New -> Blueprint**, select the
   repo. Render reads `render.yaml` and pre-fills the build/start
   commands and free plan automatically.
3. Deploy. You'll get one URL -- that's the whole app, UI and API both.

Render's free tier sleeps after 15 minutes of no traffic; the next visit
takes 30-60 seconds to wake up. That's expected on the free plan, not a
bug.

A split deployment (static frontend on Vercel, API on Render) was
considered and deliberately avoided: Vercel's serverless functions have a
hard, non-configurable 4.5MB request body limit, which would cap uploads
far below what this app is meant to handle. A single Render service has
no such limit.

## Windows / 32-bit Python notes

This project was built and hardened against a real 32-bit Python 3.8
environment along the way. A few dependency choices reflect that:

- `pandas==2.0.3` -- the last version supporting Python 3.8.
- `numpy<2.0` -- matches that pandas version.
- Plain `uvicorn`, not `uvicorn[standard]` -- the `[standard]` extra pulls
  in `httptools`, which has no 32-bit Windows wheel and fails to build
  without a C++ compiler.
- `pyarrow` (Parquet support) is optional and commented out in
  `requirements.txt` -- it has no 32-bit Windows wheels at all. Uploading a
  `.parquet` file without it gives a clear error message rather than a
  crash.
- `analysis.py` deliberately avoids `pd.cut()` / `pd.qcut()` for binning
  (used in time-series downsampling and mutual-information discretization).
  Both build a pandas `IntervalTree` internally, which has a confirmed bug
  on 32-bit pandas builds once bin counts get non-trivial
  (`Cannot cast array data from dtype('int64') to dtype('int32')`).
  Binning is done by hand with `numpy.digitize()` instead.
- PCA and KMeans clustering are hand-written with plain numpy rather than
  scikit-learn, which has a history of dropping 32-bit Windows wheel
  support.

If you're on 64-bit Python, none of this affects you -- it's purely
defensive for the 32-bit case.

## Troubleshooting

If an upload returns "Analysis failed," the terminal running the backend
prints the full traceback (not just the one-line message shown in the
browser) -- check there first for the exact file and line number.

## Extending it

- **Bigger files:** analysis runs synchronously in the request. For files
  much larger than ~50-100MB, move `build_report()` into a background job
  (Celery/RQ + Redis) and poll for the result instead.
- **More formats:** `load_dataframe()` in `analysis.py` is the one place
  to add new format support.
- **Auth:** there is none. Anyone who can reach the deployed URL can
  upload and analyze files. Add auth middleware before sharing it widely.
- **Persistence:** nothing is saved -- every upload is analyzed fresh in
  memory and discarded. Add a database layer for history or shareable
  report links.

## Limits worth knowing

- Max upload size: 100MB (`MAX_FILE_SIZE_MB` in `main.py`).
- Correlation, PCA, and mutual-information computations sample down to a
  capped number of rows on very large datasets to keep response times
  reasonable (see `max_rows_for_corr`, `max_sample` in `analysis.py`).
- ML readiness caps the number of target candidates and feature columns
  considered per request, for the same reason.
