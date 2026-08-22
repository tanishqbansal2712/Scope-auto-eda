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
    if len(ts) < 5:
        return []

    results = []
    for col in numeric_cols[:6]:  # cap to keep payload/computation reasonable
        series = ts[[datetime_col, col]].dropna()
        if len(series) < 5:
            continue
        # downsample to at most max_points via binning if needed.
        # Deliberately NOT using pd.cut() here: it builds a pandas
        # IntervalIndex/IntervalTree internally, which has a known bug on
        # 32-bit pandas builds ("Cannot cast array data from dtype('int64')
        # to dtype('int32')") once the bin count gets non-trivial. Binning
        # by hand with numpy on the integer timestamp avoids that code path
        # entirely and works identically on every platform.
        if len(series) > max_points:
            ts_numeric = series[datetime_col].to_numpy().astype("datetime64[ns]").astype("int64")
            lo, hi = ts_numeric.min(), ts_numeric.max()
            if hi == lo:
                trend_dates = [str(series[datetime_col].iloc[0])]
                trend_values = [_safe(series[col].mean())]
            else:
                edges = np.linspace(lo, hi, max_points + 1)
                bin_idx = np.clip(np.digitize(ts_numeric, edges[1:-1]), 0, max_points - 1)
                grouped = series[col].groupby(bin_idx).mean()
                bin_centers = (edges[:-1] + edges[1:]) / 2
                # explicit unit="ns" — pandas' default interpretation of a
                # bare int64 array differs across versions/datetime
                # resolutions, so this is spelled out rather than relied on.
                trend_dates = pd.to_datetime(bin_centers[grouped.index].astype("int64"), unit="ns").astype(str).tolist()
                trend_values = _safe_list(grouped)
        else:
            trend_dates = series[datetime_col].astype(str).tolist()
            trend_values = _safe_list(series[col])

        dow = series.copy()
        dow["_dow"] = dow[datetime_col].dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg = dow.groupby("_dow", observed=True)[col].mean().reindex(dow_order)

        moy = series.copy()
        moy["_moy"] = moy[datetime_col].dt.month_name()
        moy_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        moy_avg = moy.groupby("_moy", observed=True)[col].mean().reindex(moy_order)

        results.append({
            "column": str(col),
            "trend": {"dates": trend_dates, "values": trend_values},
            "day_of_week": {"labels": dow_order, "values": _safe_list(dow_avg)},
            "month_of_year": {"labels": moy_order, "values": _safe_list(moy_avg)},
        })

    return {"datetime_column": str(datetime_col), "series": results}


# ---------------------------------------------------------------------------
# ML-readiness — target detection, feature importance, class imbalance,
# split preview, and a composite readiness score. Deliberately avoids
# scikit-learn (see the scipy/sklearn note near the top of this file) —
# feature importance here is mutual information computed from scratch via
# histogram binning, not a trained model. It's a fast, dependency-free proxy
# for "which columns relate to this target", not a substitute for actually
# training something.
# ---------------------------------------------------------------------------

def _discretize(series: pd.Series, is_numeric: bool, bins: int = 10) -> np.ndarray:
    """
    Bins a series into integer codes for mutual-information computation.
    Deliberately avoids pd.cut() / pd.qcut(): both build a pandas
    IntervalIndex/IntervalTree internally (they share the same
    _bins_to_cuts code path), which has a known "Cannot cast array data
    from dtype('int64') to dtype('int32')" bug on 32-bit pandas builds.
    Quantile binning is done by hand with numpy instead, which never
    touches that code path. Returns a plain numpy int array; -1 marks
    rows that couldn't be assigned a bin (e.g. missing values).
    """
    if is_numeric:
        values = series.to_numpy(dtype="float64")
        valid = ~np.isnan(values)
        codes = np.full(len(values), -1, dtype=np.int64)
        if valid.sum() < 2:
            return codes
        quantile_pts = np.linspace(0, 100, bins + 1)
        edges = np.unique(np.percentile(values[valid], quantile_pts))
        if len(edges) < 2:
            codes[valid] = 0  # constant column — everything in one bin
            return codes
        codes[valid] = np.digitize(values[valid], edges[1:-1])
        return codes
    return series.astype("category").cat.codes.to_numpy()


def _mutual_information(a: pd.Series, b: pd.Series, a_numeric: bool, b_numeric: bool, bins: int = 10) -> float:
    """
    Normalized mutual information (symmetric uncertainty, roughly 0-1) between
    two columns. Rows where either value is missing are dropped first.
    """
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(paired) < 5:
        return 0.0
    ca = _discretize(paired["a"], a_numeric, bins)
    cb = _discretize(paired["b"], b_numeric, bins)
    valid = (ca >= 0) & (cb >= 0)
    ca, cb = ca[valid], cb[valid]
    if len(ca) < 5 or len(np.unique(ca)) < 2 or len(np.unique(cb)) < 2:
        return 0.0

    joint = pd.crosstab(ca, cb).values.astype(float)
    n = joint.sum()
    if n == 0:
        return 0.0
    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px * py)
        term = pxy * np.log(ratio)
    term = np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
    mi = term.sum()

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    hx, hy = entropy(px.flatten()), entropy(py.flatten())
    if hx + hy == 0:
        return 0.0
    nmi = 2 * mi / (hx + hy)
    return float(np.clip(nmi, 0, 1))


def _ml_readiness(df: pd.DataFrame, columns_report: List[Dict[str, Any]], overview: Dict[str, Any],
                   data_quality: Dict[str, Any], max_targets: int = 8, max_features: int = 20,
                   max_sample_rows: int = 3000) -> Dict[str, Any]:
    n_rows = overview["n_rows"]
    by_name = {c["name"]: c for c in columns_report}

    # Mutual information cost scales with rows; sample for speed on big
    # datasets the same way correlation/PCA do elsewhere in this file.
    mi_df = df.sample(max_sample_rows, random_state=42) if n_rows > max_sample_rows else df

    # Usable feature columns: numeric or categorical/boolean, not flagged as
    # an ID-like or free-text leakage risk.
    feature_cols = [
        c["name"] for c in columns_report
        if c["dtype"] in ("numeric", "categorical", "boolean")
        and c.get("cardinality_flag") != "id_like"
    ][:max_features]

    # Target candidates: numeric columns (regression) with real variance,
    # plus categorical/boolean columns (classification) with 2-20 classes
    # and no ID-like flag.
    target_candidates = []
    for c in columns_report:
        name = c["name"]
        if c["dtype"] == "numeric" and df[name].std(skipna=True) > 0:
            target_candidates.append((name, "regression"))
        elif c["dtype"] in ("categorical", "boolean") and c.get("cardinality_flag") != "id_like" and 2 <= c["unique"] <= 20:
            target_candidates.append((name, "classification"))
    target_candidates = target_candidates[:max_targets]

    targets_out = []
    for target_name, task in target_candidates:
        target_is_numeric = task == "regression"
        importances = []
        for feat in feature_cols:
            if feat == target_name:
                continue
            feat_is_numeric = by_name[feat]["dtype"] == "numeric"
            score = _mutual_information(mi_df[feat], mi_df[target_name], feat_is_numeric, target_is_numeric)
            importances.append({"feature": feat, "importance": round(score, 4)})
        importances.sort(key=lambda x: x["importance"], reverse=True)

        entry = {
            "target": target_name,
            "task": task,
            "top_features": importances[:10],
        }

        if task == "classification":
            vc = df[target_name].value_counts(dropna=True)
            total = vc.sum()
            class_dist = [{"value": str(idx), "count": int(cnt), "pct": _safe(round(cnt / total * 100, 1))} for idx, cnt in vc.items()]
            majority_pct = class_dist[0]["pct"] if class_dist else 0
            entry["class_distribution"] = class_dist
            entry["imbalanced"] = bool(majority_pct > 70)
            entry["majority_pct"] = majority_pct

        targets_out.append(entry)

    # Train/test split preview — generic, not target-specific.
    split_preview = [
        {"train_pct": pct, "train_rows": int(round(n_rows * pct / 100)), "test_rows": n_rows - int(round(n_rows * pct / 100))}
        for pct in (70, 80, 90)
    ]

    # Composite readiness score — a heuristic 0-100 blend of the data-quality
    # signals already computed elsewhere in this file, plus a rows-vs-columns
    # adequacy check. This is a rule-of-thumb summary for a human to skim,
    # not a rigorous statistic.
    missing_pct = overview["missing_pct"]
    dup_pct = (overview["duplicate_rows"] / n_rows * 100) if n_rows else 0
    n_mismatches = len(data_quality.get("type_mismatches", []))
    n_flags = len(data_quality.get("cardinality_flags", []))
    numeric_outlier_pcts = [
        c["outliers"]["pct"] for c in columns_report
        if c["dtype"] == "numeric" and c.get("outliers")
    ]
    avg_outlier_pct = sum(numeric_outlier_pcts) / len(numeric_outlier_pcts) if numeric_outlier_pcts else 0
    rows_per_col = n_rows / overview["n_cols"] if overview["n_cols"] else 0

    components = [
        {"label": "Completeness", "score": round(max(0, 100 - missing_pct * 2)), "note": f"{missing_pct}% missing overall"},
        {"label": "Uniqueness", "score": round(max(0, 100 - dup_pct * 3)), "note": f"{overview['duplicate_rows']} duplicate rows"},
        {"label": "Type consistency", "score": round(max(0, 100 - n_mismatches * 15)), "note": f"{n_mismatches} type mismatch warning(s)"},
        {"label": "Feature quality", "score": round(max(0, 100 - n_flags * 10)), "note": f"{n_flags} ID-like/high-cardinality flag(s)"},
        {"label": "Outlier prevalence", "score": round(max(0, 100 - avg_outlier_pct * 5)), "note": f"{round(avg_outlier_pct,1)}% avg outliers in numeric columns"},
        {"label": "Sample size", "score": round(min(100, (rows_per_col / 20) * 100)), "note": f"~{round(rows_per_col)} rows per column"},
    ]
    overall_score = round(sum(c["score"] for c in components) / len(components))

    return {
        "readiness_score": overall_score,
        "score_breakdown": components,
        "targets": targets_out,
        "split_preview": split_preview,
        "feature_columns_considered": feature_cols,
    }


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, max_rows_for_corr: int = 200_000) -> Dict[str, Any]:
    n_rows, n_cols = df.shape

    overview = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
        "missing_pct": round(float(df.isna().sum().sum()) / (n_rows * n_cols) * 100, 2) if n_rows and n_cols else 0,
    }

    columns_report = []
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        series = df[col]
        dtype = _classify(series)
        missing = int(series.isna().sum())
        col_info: Dict[str, Any] = {
            "name": str(col),
            "dtype": dtype,
            "pandas_dtype": str(series.dtype),
            "missing": missing,
            "missing_pct": round(missing / n_rows * 100, 2) if n_rows else 0,
            "unique": int(series.nunique(dropna=True)),
        }
        # cardinality flag is finalized after per-type processing below,
        # since text columns need avg string length to distinguish
        # short ID-like tokens from long free-text notes/comments.

        if dtype == "numeric":
            numeric_cols.append(col)
            desc = series.describe()
            col_info["stats"] = {
                "mean": _safe(desc.get("mean")),
                "std": _safe(desc.get("std")),
                "min": _safe(desc.get("min")),
                "q25": _safe(desc.get("25%")),
                "median": _safe(desc.get("50%")),
                "q75": _safe(desc.get("75%")),
                "max": _safe(desc.get("max")),
                "skew": _safe(series.skew()),
                "kurtosis": _safe(series.kurtosis()),
            }
            col_info["distribution_label"] = _distribution_label(
                col_info["stats"]["skew"], col_info["stats"]["kurtosis"]
            )
            col_info["normality_test"] = _normality_test(series)
            col_info["imputation_suggestion"] = _imputation_suggestion(
                dtype, col_info["missing_pct"], skew=_safe(series.skew())
            )
            # histogram — bin range is based on the IQR outlier bounds (the
            # same definition used below for outlier flagging) rather than a
            # fixed percentile, since a fixed percentile like 1st/99th still
            # breaks once the outlier fraction exceeds that percentile (e.g.
            # 2% of rows are extreme outliers). Values outside the range are
            # clipped into the edge bins so every row is still counted.
            clean = series.dropna()
            q1, q3 = desc.get("25%"), desc.get("75%")
            iqr = q3 - q1 if (q1 is not None and q3 is not None) else None
            if len(clean) > 0:
                n_bins = min(30, max(5, int(np.sqrt(len(clean)))))
                if iqr and iqr > 0:
                    lo = max(q1 - 1.5 * iqr, clean.min())
                    hi = min(q3 + 1.5 * iqr, clean.max())
                else:
                    lo, hi = clean.min(), clean.max()
                if lo == hi:
                    lo, hi = clean.min(), clean.max()
                if lo == hi:
                    hi = lo + 1  # constant column — avoid a zero-width range
                edges = np.linspace(lo, hi, n_bins + 1)
                counts, edges = np.histogram(clean.clip(lo, hi), bins=edges)
                col_info["histogram"] = {
                    "counts": _safe_list(counts),
                    "bin_edges": _safe_list(edges),
                }
                # outliers via IQR
                lower, upper = (q1 - 1.5 * iqr, q3 + 1.5 * iqr) if iqr else (clean.min(), clean.max())
                outliers = clean[(clean < lower) | (clean > upper)]
                col_info["outliers"] = {
                    "count": int(len(outliers)),
                    "pct": round(len(outliers) / len(clean) * 100, 2),
                    "lower_bound": _safe(lower),
                    "upper_bound": _safe(upper),
                }

        elif dtype in ("categorical", "boolean"):
            categorical_cols.append(col)
            vc = series.value_counts(dropna=True).head(15)
            col_info["top_values"] = [
                {"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()
            ]
            col_info["imputation_suggestion"] = _imputation_suggestion(dtype, col_info["missing_pct"])
            mismatch = _detect_type_mismatch(series, dtype)
            if mismatch:
                col_info["type_mismatch"] = mismatch

        elif dtype == "text":
            lengths = series.dropna().astype(str).str.len()
            col_info["text_stats"] = {
                "avg_length": _safe(lengths.mean()) if len(lengths) else None,
                "max_length": _safe(lengths.max()) if len(lengths) else None,
            }
            col_info["imputation_suggestion"] = _imputation_suggestion(dtype, col_info["missing_pct"])
            mismatch = _detect_type_mismatch(series, dtype)
            if mismatch:
                col_info["type_mismatch"] = mismatch

        elif dtype == "datetime":
            try:
                parsed = pd.to_datetime(series, errors="coerce")
                col_info["datetime_stats"] = {
                    "min": _safe(parsed.min()),
                    "max": _safe(parsed.max()),
                }
            except Exception:
                pass
            col_info["imputation_suggestion"] = _imputation_suggestion(dtype, col_info["missing_pct"])

        avg_len = col_info.get("text_stats", {}).get("avg_length") if "text_stats" in col_info else None
        col_info["cardinality_flag"] = _cardinality_flag(col_info["unique"], n_rows, dtype, avg_length=avg_len)

        columns_report.append(col_info)

    # Correlation matrix (numeric columns only, capped for perf)
    correlation = None
    if len(numeric_cols) >= 2:
        sample_df = df[numeric_cols]
        if n_rows > max_rows_for_corr:
            sample_df = sample_df.sample(max_rows_for_corr, random_state=42)
        corr = sample_df.corr(numeric_only=True).round(3)
        correlation = {
            "columns": [str(c) for c in corr.columns],
            "matrix": [[_safe(v) for v in row] for row in corr.values],
        }

    # Strongest correlated pairs (excluding self-pairs)
    top_correlations = []
    if correlation:
        cols = correlation["columns"]
        mat = correlation["matrix"]
        seen = set()
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = mat[i][j]
                if v is not None:
                    pairs.append((abs(v), cols[i], cols[j], v))
        pairs.sort(reverse=True)
        top_correlations = [
            {"col_a": a, "col_b": b, "correlation": v} for _, a, b, v in pairs[:8]
        ]

    # ---- Data quality layer ----
    missingness = _missingness_layer(df)
    type_mismatches = [
        {"column": c["name"], **c["type_mismatch"]}
        for c in columns_report
        if c.get("type_mismatch")
    ]
    cardinality_flags = [
        {"column": c["name"], "unique": c["unique"], "unique_pct": round(c["unique"] / n_rows * 100, 1) if n_rows else 0,
         "flag": c["cardinality_flag"]}
        for c in columns_report
        if c["cardinality_flag"] != "none"
    ]
    dup_mask = df.duplicated(keep=False)
    duplicate_sample = json.loads(
        df[dup_mask].head(10).to_json(orient="records", date_format="iso")
    ) if dup_mask.any() else []

    data_quality = {
        "missingness": missingness,
        "type_mismatches": type_mismatches,
        "cardinality_flags": cardinality_flags,
        "duplicate_sample": duplicate_sample,
    }

    # ---- Deeper statistics ----
    chi_square = _chi_square_tests(df, categorical_cols)
    pca_clusters = _pca_and_clusters(df, numeric_cols)
    time_series = _time_series_summary(df, numeric_cols)

    deeper_statistics = {
        "scipy_available": _SCIPY_AVAILABLE,
        "chi_square_tests": chi_square,
        "pca": pca_clusters,
        "time_series": time_series if time_series else None,
    }

    # ---- ML readiness ----
    ml_readiness = _ml_readiness(df, columns_report, overview, data_quality)

    return {
        "overview": overview,
        "columns": columns_report,
        "numeric_columns": [str(c) for c in numeric_cols],
        "categorical_columns": [str(c) for c in categorical_cols],
        "correlation": correlation,
        "top_correlations": top_correlations,
        "data_quality": data_quality,
        "deeper_statistics": deeper_statistics,
        "ml_readiness": ml_readiness,
        "preview_rows": json.loads(df.head(10).to_json(orient="records", date_format="iso")),
    }
