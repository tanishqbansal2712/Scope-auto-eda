"""
analysis.py
Core automated EDA engine. Takes a pandas DataFrame and produces a fully
JSON-serializable report: dataset overview, per-column stats, missing data,
correlations, distributions, categorical breakdowns, and outliers.
"""

import io
import json
import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# scipy is used for the Shapiro-Wilk normality test and the chi-square test
# of independence. It's optional: if it isn't installed (or fails to build
# on a fragile environment), those two specific features are skipped rather
# than breaking the whole app. PCA and clustering below are implemented with
# plain numpy for the same reason — no scikit-learn dependency needed.
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# File loading — auto-detects format from filename + tries to be forgiving
# ---------------------------------------------------------------------------

def load_dataframe(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    name = filename.lower()
    buf = io.BytesIO(raw_bytes)

    if name.endswith((".csv", ".txt", ".tsv")):
        try:
            if name.endswith(".tsv"):
                df = pd.read_csv(buf, sep="\t")
            else:
                df = pd.read_csv(buf, sep=None, engine="python")
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    elif name.endswith(".json"):
        buf.seek(0)
        try:
            df = pd.read_json(buf)
        except ValueError:
            buf.seek(0)
            data = json.load(buf)
            df = pd.json_normalize(data)
    elif name.endswith(".parquet"):
        try:
            df = pd.read_parquet(buf)
        except ImportError:
            raise ValueError(
                "Parquet support needs the 'pyarrow' package, which isn't installed. "
                "Run: pip install pyarrow  (or use CSV/Excel/JSON instead)."
            )
    else:
        # last-resort attempt: sniff as csv
        buf.seek(0)
        df = pd.read_csv(buf, sep=None, engine="python")

    # Drop fully-empty unnamed index columns some exports leave behind
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed and df[unnamed].isna().all().all():
        df = df.drop(columns=unnamed)

    # Normalize every column name to a plain string right away. Excel files
    # in particular can have non-string column headers — a date-formatted
    # header cell (e.g. a wide "revenue by day" sheet) gets parsed by
    # openpyxl/pandas as an actual Timestamp object, not text. Left as-is,
    # that caused a KeyError deep in the ML-readiness code, which stringifies
    # column names for display and then used those strings to look the
    # column back up in the DataFrame — a Timestamp column doesn't match its
    # own string form as a lookup key. Normalizing here, once, at the source
    # means every function downstream can safely assume string column names.
    df.columns = [str(c) for c in df.columns]

    return df


# ---------------------------------------------------------------------------
# JSON-safety helpers — numpy/pandas types and NaN/Inf aren't JSON-safe
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _safe_list(seq) -> List[Any]:
    return [_safe(v) for v in seq]


# ---------------------------------------------------------------------------
# Column type classification
# ---------------------------------------------------------------------------

def _classify(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    is_stringlike = (
        series.dtype == object
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
    )
    if is_stringlike:
        # try datetime parse on a sample
        sample = series.dropna().head(50)
        if len(sample) > 0:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(sample, errors="raise")
                return "datetime"
            except Exception:
                pass
        nunique = series.nunique(dropna=True)
        if nunique <= max(20, int(0.05 * len(series))):
            return "categorical"
        return "text"
    return "other"


# ---------------------------------------------------------------------------
# Data quality layer helpers
# ---------------------------------------------------------------------------

def _detect_type_mismatch(series: pd.Series, dtype: str):
    """
    Flags columns that LOOK numeric but got classified as text/categorical
    because a handful of stray non-numeric values are mixed in
    (e.g. "42", "37", "N/A", "51", "unknown").
    Returns None if no mismatch, else a dict with details.
    """
    if dtype not in ("text", "categorical"):
        return None
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return None
    numeric_parsed = pd.to_numeric(clean, errors="coerce")
    parseable = numeric_parsed.notna()
    parseable_pct = parseable.mean() * 100
    # Only flag when MOST values are numeric but not all — a fully numeric
    # column would already be classified "numeric", and a fully text column
    # isn't a mismatch, it's just text.
    if 60 <= parseable_pct < 100:
        bad_values = clean[~parseable].unique()[:5].tolist()
        return {
            "looks_like": "numeric",
            "parseable_pct": round(parseable_pct, 1),
            "example_bad_values": bad_values,
        }
    return None


def _imputation_suggestion(dtype: str, missing_pct: float, skew: float = None) -> str:
    if missing_pct == 0:
        return "No missing values — no imputation needed."
    if missing_pct > 40:
        return f"{missing_pct:.0f}% missing — consider dropping this column rather than imputing."
    if dtype == "numeric":
        if skew is not None and abs(skew) > 1:
            return "Skewed distribution — impute with median (robust to outliers)."
        return "Roughly symmetric — impute with mean, or median if unsure."
    if dtype in ("categorical", "boolean"):
        return "Impute with the mode (most frequent value), or add an explicit 'Missing' category."
    if dtype == "datetime":
        return "Consider forward-fill / backward-fill, or a placeholder date with a missing-flag column."
    return "Impute with a placeholder value, or add a boolean 'was_missing' flag column."


def _cardinality_flag(unique: int, n_rows: int, dtype: str, avg_length: float = None) -> str:
    # Only categorical/text columns benefit from this check. Continuous
    # numeric columns (salary, measurements, etc.) are naturally close to
    # 100% unique — that's normal, not an ID-like trap — and datetime
    # columns are excluded for the same reason.
    if n_rows == 0 or dtype not in ("categorical", "text"):
        return "none"
    ratio = unique / n_rows
    # Long free-text fields (notes, comments) are also naturally near-100%
    # unique without being "ID-like" — only flag short, token-shaped values
    # (codes, usernames, SKUs) as a likely identifier column.
    is_short = avg_length is None or avg_length <= 25
    if ratio > 0.95 and unique > 10 and is_short:
        return "id_like"
    if unique > 50 or (ratio > 0.5 and unique > 20):
        return "high_cardinality"
    return "none"


def _missingness_layer(df: pd.DataFrame, max_sample_rows: int = 150) -> Dict[str, Any]:
    n_rows = len(df)
    cols_with_missing = [c for c in df.columns if df[c].isna().any()]

    layer: Dict[str, Any] = {
        "columns_with_missing": [str(c) for c in cols_with_missing],
    }

    # Nullity matrix — a sample of rows so the payload stays small even on
    # big datasets, shown as a heatmap on the frontend (missing = 1).
    if cols_with_missing and n_rows > 0:
        sample = df[cols_with_missing]
        if n_rows > max_sample_rows:
            sample = sample.sample(max_sample_rows, random_state=42).sort_index()
        layer["matrix"] = {
            "columns": [str(c) for c in cols_with_missing],
            "rows": sample.isna().astype(int).values.tolist(),
        }

    # Missingness correlation — do columns tend to be missing together?
    # Only meaningful with 2+ columns that actually have missing values.
    if len(cols_with_missing) >= 2:
        null_indicator = df[cols_with_missing].isna().astype(int)
        # a column that's always/never missing has zero variance and
        # produces NaN correlations — drop those before correlating
        varying = [c for c in cols_with_missing if null_indicator[c].nunique() > 1]
        if len(varying) >= 2:
            corr = null_indicator[varying].corr().round(3)
            layer["correlation"] = {
                "columns": [str(c) for c in corr.columns],
                "matrix": [[_safe(v) for v in row] for row in corr.values],
            }

    return layer


# ---------------------------------------------------------------------------
# Deeper statistics — distribution shape, normality, chi-square, PCA, clustering
# ---------------------------------------------------------------------------

def _distribution_label(skew: float, kurtosis: float) -> str:
    """Plain-language read on shape from skew/kurtosis — no test required."""
    if skew is None:
        return "unknown"
    if abs(skew) < 0.5 and (kurtosis is None or abs(kurtosis) < 1):
        return "approximately normal"
    if skew >= 0.5:
        label = "right-skewed"
    elif skew <= -0.5:
        label = "left-skewed"
    else:
        label = "roughly symmetric"
    if kurtosis is not None and kurtosis > 3:
        label += ", heavy-tailed"
    elif kurtosis is not None and kurtosis < -1:
        label += ", flat-topped"
    return label


def _normality_test(series: pd.Series, max_sample: int = 5000) -> Dict[str, Any]:
    """
    Shapiro-Wilk test for normality. Requires scipy; returns None if
    unavailable. Sampled for large columns since Shapiro-Wilk loses
    reliability (and gets slow) well past a few thousand points.
    """
    if not _SCIPY_AVAILABLE:
        return None
    clean = series.dropna()
    if len(clean) < 3:
        return None
    sample = clean.sample(max_sample, random_state=42) if len(clean) > max_sample else clean
    try:
        # Explicit float64 cast avoids a numpy safe-casting error that shows
        # up only on 32-bit Windows: numpy's platform-default int there is
        # int32, but pandas integer columns are int64, and scipy's internal
        # buffers can end up sized for the former while fed the latter.
        # Casting to float64 sidesteps the int-width mismatch entirely.
        stat, p_value = _scipy_stats.shapiro(sample.to_numpy(dtype=np.float64))
    except Exception:
        return None
    return {
        "statistic": _safe(stat),
        "p_value": _safe(p_value),
        "is_normal": bool(p_value > 0.05),
        "sampled": len(sample) < len(clean),
        "sample_size": int(len(sample)),
    }


def _chi_square_tests(df: pd.DataFrame, categorical_cols: List[str], max_categories: int = 20, max_pairs: int = 15) -> List[Dict[str, Any]]:
    """
    Chi-square test of independence for pairs of categorical columns.
    Skips columns with too many categories (contingency tables get sparse
    and the test becomes unreliable) and caps the number of pairs tested
    so this stays fast on wide datasets.
    """
    if not _SCIPY_AVAILABLE:
        return []
    eligible = [c for c in categorical_cols if df[c].nunique(dropna=True) <= max_categories and df[c].nunique(dropna=True) >= 2]
    results = []
    pairs_tested = 0
    for i in range(len(eligible)):
        if pairs_tested >= max_pairs:
            break
        for j in range(i + 1, len(eligible)):
            if pairs_tested >= max_pairs:
                break
            col_a, col_b = eligible[i], eligible[j]
            try:
                contingency = pd.crosstab(df[col_a], df[col_b])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                # float64 cast — see the note in _normality_test above about
                # why raw pandas/numpy int arrays can trip a scipy internal
                # cast on 32-bit Windows.
                chi2, p_value, dof, _ = _scipy_stats.chi2_contingency(contingency.to_numpy(dtype=np.float64))
                results.append({
                    "col_a": str(col_a),
                    "col_b": str(col_b),
                    "chi2": _safe(chi2),
                    "p_value": _safe(p_value),
                    "significant": bool(p_value < 0.05),
                    "dof": int(dof),
                })
                pairs_tested += 1
            except Exception:
                continue
    results.sort(key=lambda r: r["p_value"])
    return results


def _pca_and_clusters(df: pd.DataFrame, numeric_cols: List[str], max_sample: int = 2000, max_k: int = 8) -> Dict[str, Any]:
    """
    2-component PCA via numpy SVD (no scikit-learn needed) plus a KMeans
    clustering preview implemented from scratch with numpy — both mean-
    imputed and standardized first. Returns the 2D projection, explained
    variance, an elbow curve (inertia per k), and cluster labels for the
    chosen k so the frontend can color the PCA scatter by cluster.
    """
    if len(numeric_cols) < 2:
        return None

    data = df[numeric_cols].copy()
    if len(data) > max_sample:
        data = data.sample(max_sample, random_state=42)
    data = data.apply(lambda col: col.fillna(col.mean()))
    data = data.loc[:, data.std(numeric_only=True) > 0]  # drop constant columns
    if data.shape[1] < 2 or len(data) < 4:
        return None

    X = data.values.astype(float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    Xs = (X - mean) / std

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    explained_var = (S ** 2) / np.sum(S ** 2)
    n_components = min(2, Vt.shape[0])
    projection = Xs @ Vt[:n_components].T  # (n_samples, n_components)

    # KMeans from scratch (Lloyd's algorithm), run for k=1..max_k for an
    # elbow curve, then pick k via the "distance to the chord" elbow
    # heuristic and keep that run's labels for the scatter coloring.
    rng = np.random.RandomState(42)

    def kmeans_plusplus_init(X, k):
        n = len(X)
        centers = np.empty((k, X.shape[1]))
        first = rng.randint(n)
        centers[0] = X[first]
        closest_sq_dist = ((X - centers[0]) ** 2).sum(axis=1)
        for i in range(1, k):
            probs = closest_sq_dist / closest_sq_dist.sum()
            next_idx = rng.choice(n, p=probs)
            centers[i] = X[next_idx]
            new_sq_dist = ((X - centers[i]) ** 2).sum(axis=1)
            closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)
        return centers

    def kmeans(X, k, n_init=5, max_iter=100):
        best_inertia, best_labels, best_centers = None, None, None
        for _ in range(n_init):
            centers = kmeans_plusplus_init(X, k)
            for _ in range(max_iter):
                dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                labels = dists.argmin(axis=1)
                new_centers = np.array([
                    X[labels == c].mean(axis=0) if np.any(labels == c) else centers[c]
                    for c in range(k)
                ])
                if np.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            inertia = dists[np.arange(len(X)), labels].sum()
            if best_inertia is None or inertia < best_inertia:
                best_inertia, best_labels, best_centers = inertia, labels, centers
        return best_inertia, best_labels, best_centers

    max_k = min(max_k, len(Xs) - 1, 10)
    if max_k < 2:
        return {
            "explained_variance": _safe_list(explained_var[:n_components]),
            "projection": [[_safe(v) for v in row] for row in projection.tolist()],
            "elbow": None,
            "clusters": None,
            "chosen_k": None,
        }

    inertias = []
    labels_by_k = {}
    for k in range(1, max_k + 1):
        inertia, labels, _ = kmeans(Xs, k)
        inertias.append(inertia)
        labels_by_k[k] = labels

    # Elbow heuristic: pick the k whose point lies furthest from the
    # straight line connecting the first and last point of the inertia
    # curve (the classic "kneedle" approximation).
    ks = np.arange(1, max_k + 1)
    p1 = np.array([ks[0], inertias[0]])
    p2 = np.array([ks[-1], inertias[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    chosen_k = 2
    if line_len > 0:
        line_unit = line_vec / line_len
        max_dist, best_k = -1, 2
        for k, inertia in zip(ks, inertias):
            p = np.array([k, inertia])
            proj_len = np.dot(p - p1, line_unit)
            proj_point = p1 + proj_len * line_unit
            dist = np.linalg.norm(p - proj_point)
            if dist > max_dist and k >= 2:
                max_dist, best_k = dist, k
        chosen_k = best_k

    chosen_labels = labels_by_k[chosen_k]

    return {
        "explained_variance": _safe_list(explained_var[:n_components]),
        "projection": [[_safe(v) for v in row] for row in projection.tolist()],
        "elbow": {"k": ks.tolist(), "inertia": _safe_list(inertias)},
        "clusters": chosen_labels.tolist(),
        "chosen_k": int(chosen_k),
    }


def _time_series_summary(df: pd.DataFrame, numeric_cols: List[str], max_points: int = 200) -> List[Dict[str, Any]]:
    """
    For the first usable datetime column, builds a lightweight trend line
    per numeric column (resampled/downsampled to a manageable number of
    points) plus a simple day-of-week and month-of-year seasonality
    breakdown — all via pandas groupby, no statsmodels dependency.
    """
    datetime_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_col = col
            break
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(50)
            if len(sample) > 0:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(sample, errors="raise")
                    datetime_col = col
                    break
                except Exception:
                    continue
    if datetime_col is None or not numeric_cols:
        return []

    ts = df[[datetime_col] + numeric_cols].copy()
    ts[datetime_col] = pd.to_datetime(ts[datetime_col], errors="coerce")
    ts = ts.dropna(subset=[datetime_col]).sort_values(datetime_col)
    if len(ts) 
