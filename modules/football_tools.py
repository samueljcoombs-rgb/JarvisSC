# worker/modal_app.py
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import modal

app = modal.App("football-research-worker")

# CatBoost (1.2.7) requires numpy < 2.0
image = (
    modal.Image.debian_slim()
    .apt_install("libgomp1")
    .pip_install(
        "pandas==2.2.2",
        "numpy==1.26.4",
        "requests==2.32.3",
        "supabase==2.6.0",
        "scikit-learn==1.5.2",
        "xgboost==2.1.3",
        "lightgbm==4.5.0",
        "catboost==1.2.7",
    )
)

SUPABASE_SECRET = modal.Secret.from_name("football-supabase")


# -----------------------------
# Helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _load_csv_from_supabase_storage(sb, bucket: str, path: str):
    import pandas as pd
    from io import BytesIO

    res = sb.storage.from_(bucket).download(path)
    if res is None:
        raise RuntimeError(f"Storage download returned None for {bucket}/{path}")
    raw = bytes(res) if isinstance(res, (bytes, bytearray)) else res
    return pd.read_csv(BytesIO(raw), low_memory=False)


def _to_dt(df):
    import pandas as pd

    if "DATE" in df.columns and "TIME" in df.columns:
        dt = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"].astype(str), errors="coerce")
        if dt.notna().any():
            return dt
    if "DATE" in df.columns:
        dt = pd.to_datetime(df["DATE"], errors="coerce")
        if dt.notna().any():
            return dt
    return pd.to_datetime(pd.Series([None] * len(df)), errors="coerce")


def _time_sort(df):
    tmp = df.copy()
    tmp["_dt"] = _to_dt(tmp)
    tmp = tmp.sort_values("_dt", na_position="last")
    return tmp.drop(columns=["_dt"])


def _time_split(df, ratio: float = 0.7):
    import numpy as np

    ratio = float(ratio or 0.7)
    ratio = max(0.5, min(ratio, 0.95))
    tmp = df.copy()
    tmp["_dt"] = _to_dt(tmp)
    tmp = tmp.sort_values("_dt", na_position="last")
    cut = int(np.floor(len(tmp) * ratio))
    train = tmp.iloc[:cut].drop(columns=["_dt"])
    test = tmp.iloc[cut:].drop(columns=["_dt"])
    return train, test


def _time_split_3way(df, train_ratio: float = 0.6, val_ratio: float = 0.2):
    import numpy as np

    train_ratio = max(0.4, min(float(train_ratio), 0.8))
    val_ratio = max(0.1, min(float(val_ratio), 0.3))
    if train_ratio + val_ratio >= 0.95:
        val_ratio = 0.2

    tmp = df.copy()
    tmp["_dt"] = _to_dt(tmp)
    tmp = tmp.sort_values("_dt", na_position="last")
    n = len(tmp)
    a = int(np.floor(n * train_ratio))
    b = int(np.floor(n * (train_ratio + val_ratio)))
    train = tmp.iloc[:a].drop(columns=["_dt"])
    val = tmp.iloc[a:b].drop(columns=["_dt"])
    test = tmp.iloc[b:].drop(columns=["_dt"])
    return train, val, test


# -----------------------------
# Risk metrics
# -----------------------------
def _max_drawdown(points: List[float]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in points:
        cum += float(p)
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return float(max_dd)


def _longest_losing_streak(points: List[float]) -> Dict[str, Any]:
    max_bets = 0
    cur_bets = 0
    worst_pl = 0.0
    cur_pl = 0.0
    for p in points:
        if p < 0:
            cur_bets += 1
            cur_pl += float(p)
            max_bets = max(max_bets, cur_bets)
            worst_pl = min(worst_pl, cur_pl)
        else:
            cur_bets = 0
            cur_pl = 0.0
    return {"bets": int(max_bets), "pl": float(worst_pl)}


def _game_level(df, pl_col: str) -> Dict[str, Any]:
    import pandas as pd

    if "ID" not in df.columns:
        return {"error": "Missing ID column"}
    d = df[df[pl_col].notna()].copy()
    d[pl_col] = pd.to_numeric(d[pl_col], errors="coerce")
    d = d[d[pl_col].notna()]
    if len(d) == 0:
        return {"unique_ids": 0, "max_dd": 0.0, "losing_streak": {"bets": 0, "pl": 0.0}}

    d["_dt"] = _to_dt(d)
    g = (
        d.groupby("ID", as_index=False)
        .agg(game_pl=(pl_col, "sum"), date=("_dt", "min"))
        .sort_values("date", na_position="last")
    )
    pts = g["game_pl"].astype(float).tolist()
    return {"unique_ids": int(len(g)), "max_dd": _max_drawdown(pts), "losing_streak": _longest_losing_streak(pts)}


# -----------------------------
# ROI
# -----------------------------
def _roi(df, pl_col: str, side: str, odds_col: Optional[str]) -> Dict[str, Any]:
    import pandas as pd

    d = df[df[pl_col].notna()].copy()
    d[pl_col] = pd.to_numeric(d[pl_col], errors="coerce")
    d = d[d[pl_col].notna()]
    n = int(len(d))
    total_pl = float(d[pl_col].sum()) if n else 0.0

    if n == 0:
        return {"rows": 0, "total_pl": 0.0, "roi": 0.0, "avg_pl": 0.0, "denom": 0.0}

    side = (side or "back").lower().strip()
    if side == "lay":
        if not odds_col or odds_col not in d.columns:
            return {"error": f"Lay ROI needs odds_col. Missing/invalid: {odds_col}"}
        odds = pd.to_numeric(d[odds_col], errors="coerce").fillna(0.0)
        liability = (odds - 1.0).clip(lower=0.0)
        denom = float(liability.sum())
        roi = (total_pl / denom) if denom > 0 else 0.0
        return {"rows": n, "total_pl": total_pl, "roi": roi, "avg_pl": total_pl / n, "denom": denom, "mode": "lay_liability"}

    denom = float(n)
    return {"rows": n, "total_pl": total_pl, "roi": total_pl / denom, "avg_pl": total_pl / denom, "denom": denom, "mode": "back_flat_1pt"}


# -----------------------------
# Market mapping (default odds/side)
# -----------------------------
def _mapping() -> Dict[str, Tuple[str, str]]:
    return {
        "SHG PL": ("lay", "HT CS Price"),
        "SHG 2+ PL": ("lay", "HT 2 Ahead Odds"),
        "LU1.5 PL": ("lay", "U1.5 Odds"),
        "LFGHU0.5 PL": ("lay", "FHGU0.5Odds"),
        "BO 2.5 PL": ("back", "O2.5 Odds"),
        "BO1.5 FHG PL": ("back", "FHGO1.5 Odds"),
        "BTTS PL": ("back", "BTTS Y Odds"),
    }


# ============================================================
# Feature selection / sheet enforcement
# ============================================================
def _safe_feature_columns(
    df,
    target_pl_col: str,
    ignored_columns: List[str],
    outcome_columns: List[str],
) -> List[str]:
    ignored_norm = {str(x).strip().lower() for x in (ignored_columns or []) if str(x).strip()}
    outcome_norm = {str(x).strip().lower() for x in (outcome_columns or []) if str(x).strip()}
    target_norm = str(target_pl_col).strip().lower()

    # hard bans from your sheet rules + common leakage
    banned_substrings = [
        " pl", "pl ", "pl_", "_pl", "p&l", "profit",
        "bet result", "result",
        "wins", "strike rate", "implied odds",
        "no games", "return", "outcome",
        "home form",  # sheet says ignore completely
    ]

    cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == target_norm:
            continue
        if cl in ignored_norm:
            continue
        if cl in outcome_norm:
            continue

        bad = False
        for s in banned_substrings:
            if s in cl:
                bad = True
                break
        if bad:
            continue

        cols.append(c)
    return cols


def _build_preprocessor(df, feature_cols: List[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    import pandas as pd

    X = df[feature_cols].copy()
    cat_cols = []
    num_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, num_cols, cat_cols


def _impute_maps(train, num_cols: List[str], cat_cols: List[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Build explicit imputation maps so distilled rules can be applied
    consistently to raw (unimputed) data.
    """
    import pandas as pd

    num_med = {}
    for c in num_cols:
        s = pd.to_numeric(train[c], errors="coerce")
        num_med[c] = float(s.median()) if s.notna().any() else 0.0

    cat_mode = {}
    for c in cat_cols:
        s = train[c].astype(str)
        # treat "nan" and empty as missing-ish, but keep simple
        s = s.replace("nan", "").replace("None", "")
        mode = s[s != ""].mode()
        cat_mode[c] = mode.iloc[0] if len(mode) else ""
    return num_med, cat_mode


# ============================================================
# ML models
# ============================================================
def _make_models(task: str):
    from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
    from xgboost import XGBRegressor, XGBClassifier
    from lightgbm import LGBMRegressor, LGBMClassifier
    from catboost import CatBoostRegressor, CatBoostClassifier

    if task == "regression":
        return {
            "ridge": Ridge(alpha=2.0, random_state=42),
            "elasticnet": ElasticNet(alpha=0.002, l1_ratio=0.2, random_state=42, max_iter=5000),
            "xgb": XGBRegressor(
                n_estimators=800, max_depth=5, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=2
            ),
            "lgbm": LGBMRegressor(
                n_estimators=1200, learning_rate=0.03, num_leaves=63,
                subsample=0.9, colsample_bytree=0.9, random_state=42
            ),
            "cat": CatBoostRegressor(
                iterations=1200, learning_rate=0.03, depth=6,
                loss_function="RMSE", random_seed=42, verbose=False
            ),
        }

    return {
        "logreg_en": LogisticRegression(
            solver="saga", penalty="elasticnet", l1_ratio=0.15, C=0.6,
            max_iter=4000, n_jobs=2
        ),
        "xgb": XGBClassifier(
            n_estimators=900, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=2, eval_metric="logloss"
        ),
        "lgbm": LGBMClassifier(
            n_estimators=1400, learning_rate=0.03, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ),
        "cat": CatBoostClassifier(
            iterations=1400, learning_rate=0.03, depth=6,
            loss_function="Logloss", random_seed=42, verbose=False
        ),
    }


def _fit_predict_pipeline(model, pre, X_train, y_train, X_eval):
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(X_eval)
        if p.ndim == 2 and p.shape[1] >= 2:
            return pipe, p[:, 1]
        return pipe, p.ravel()
    return pipe, pipe.predict(X_eval)


# ============================================================
# Rule application (supports numeric + categoricals)
# ============================================================
def _apply_rule_spec(df, spec: Dict[str, Any], num_med: Dict[str, float], cat_mode: Dict[str, Any]):
    """
    spec = {
      "numeric": [{"col":..., "min":..., "max":...}, ...],
      "categorical": [{"col":..., "in": [...]} , {"col":..., "not_in":[...]}]
    }
    """
    import pandas as pd

    d = df.copy()

    # numeric
    for cond in spec.get("numeric", []) or []:
        col = cond.get("col")
        if not col or col not in d.columns:
            continue
        s = pd.to_numeric(d[col], errors="coerce")
        s = s.fillna(num_med.get(col, 0.0))
        a = cond.get("min", None)
        b = cond.get("max", None)
        mask = pd.Series([True] * len(d), index=d.index)
        if a is not None:
            mask = mask & (s >= float(a))
        if b is not None:
            mask = mask & (s <= float(b))
        d = d[mask]

    # categorical
    for cond in spec.get("categorical", []) or []:
        col = cond.get("col")
        if not col or col not in d.columns:
            continue
        s = d[col].astype(str)
        fill = str(cat_mode.get(col, ""))
        s = s.replace("nan", "").replace("None", "")
        s = s.mask((s == "") | (s.isna()), fill)

        if "in" in cond and cond["in"] is not None:
            allowed = {str(x) for x in (cond["in"] or [])}
            d = d[s.isin(allowed)]
        if "not_in" in cond and cond["not_in"] is not None:
            banned = {str(x) for x in (cond["not_in"] or [])}
            d = d[~s.isin(banned)]

    return d


# ============================================================
# NEW: Distill with categoricals (tree over transformed features)
# ============================================================
def _feature_name_to_ohe(original_cat_cols: List[str], feat_name: str) -> Optional[Tuple[str, str]]:
    """
    OneHotEncoder feature names typically look like:
      cat__LEAGUE_Premier League
    Return (col, category) if matches.
    """
    if not feat_name.startswith("cat__"):
        return None
    rest = feat_name[len("cat__"):]
    # split on first "_" because categories themselves can contain underscores
    if "_" not in rest:
        return None
    col, cat = rest.split("_", 1)
    if col in original_cat_cols:
        return (col, cat)
    return None


def _extract_rule_specs_from_tree(tree, feature_names: List[str], cat_cols: List[str]) -> List[Dict[str, Any]]:
    """
    Extract leaf rules. Translate:
      num__X threshold -> numeric ranges
      cat__COL_VALUE threshold -> categorical in/not_in
    """
    import numpy as np
    from sklearn.tree import _tree

    t = tree.tree_
    rules = []

    def walk(node: int,
             num_bounds: Dict[str, Tuple[Optional[float], Optional[float]]],
             cat_in: Dict[str, Optional[set]],
             cat_not: Dict[str, set]):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            val = float(t.value[node].ravel()[0])
            n = int(t.n_node_samples[node])

            numeric = []
            for col, (lo, hi) in num_bounds.items():
                numeric.append({"col": col, "min": lo, "max": hi})
            numeric.sort(key=lambda x: x["col"])

            categorical = []
            for col in sorted(set(list(cat_in.keys()) + list(cat_not.keys()))):
                inc = cat_in.get(col, None)
                exc = cat_not.get(col, set())
                if inc is not None:
                    categorical.append({"col": col, "in": sorted(list(inc))})
                if exc:
                    categorical.append({"col": col, "not_in": sorted(list(exc))})

            rules.append({"spec": {"numeric": numeric, "categorical": categorical}, "leaf_value": val, "n_node_samples": n})
            return

        feat = feature_names[int(t.feature[node])]
        thr = float(t.threshold[node])

        # NUMERIC feature
        if feat.startswith("num__"):
            col = feat[len("num__"):]
            lo, hi = num_bounds.get(col, (None, None))

            # left: x <= thr
            nb_left = dict(num_bounds)
            nb_left[col] = (lo, thr if hi is None else min(hi, thr))
            walk(int(t.children_left[node]), nb_left, cat_in, cat_not)

            # right: x > thr
            nb_right = dict(num_bounds)
            nb_right[col] = (thr if lo is None else max(lo, thr), hi)
            walk(int(t.children_right[node]), nb_right, cat_in, cat_not)
            return

        # CATEGORICAL one-hot feature
        ohe = _feature_name_to_ohe(cat_cols, feat)
        if ohe is None:
            # unknown feature; just walk both sides without constraints
            walk(int(t.children_left[node]), num_bounds, cat_in, cat_not)
            walk(int(t.children_right[node]), num_bounds, cat_in, cat_not)
            return

        col, cat = ohe

        # for one-hot, values are 0/1. Tree splits usually around 0.5.
        # left: x <= thr  (often means NOT this category)
        ci_left = {k: (None if v is None else set(v)) for k, v in cat_in.items()}
        cn_left = {k: set(v) for k, v in cat_not.items()}
        if thr >= 0.5:
            cn_left.setdefault(col, set()).add(cat)
        walk(int(t.children_left[node]), num_bounds, ci_left, cn_left)

        # right: x > thr (often means IS this category)
        ci_right = {k: (None if v is None else set(v)) for k, v in cat_in.items()}
        cn_right = {k: set(v) for k, v in cat_not.items()}
        if thr >= 0.5:
            # must be this cat
            existing = ci_right.get(col, None)
            if existing is None:
                ci_right[col] = {cat}
            else:
                existing.add(cat)
                ci_right[col] = existing
        walk(int(t.children_right[node]), num_bounds, ci_right, cn_right)

    walk(0, {}, {}, {})
    rules.sort(key=lambda r: (r["leaf_value"], r["n_node_samples"]), reverse=True)
    return rules


def _distill_best_model_to_rules(
    best_kind: str,
    best_model_name: str,
    model_obj,
    pre,
    train,
    val,
    test,
    feature_cols: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    pl_col: str,
    side: str,
    odds_col: str,
) -> Dict[str, Any]:
    """
    - Fit base model on TRAIN only.
    - Generate a score for TRAIN (reg: predicted PL, clf: probability of PL>0)
    - Fit surrogate shallow DecisionTreeRegressor on the TRANSFORMED feature matrix.
    - Extract explicit rules including categoricals, evaluate on TRAIN/VAL/TEST.
    """
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor

    X_train = train[feature_cols]
    X_val = val[feature_cols]
    X_test = test[feature_cols]

    # base scores (train-only fit)
    if best_kind == "ml_reg":
        y_train = train[pl_col].astype(float).values
        base_pipe, s_train = _fit_predict_pipeline(model_obj, pre, X_train, y_train, X_train)
    else:
        y_train = (train[pl_col].astype(float).values > 0).astype(int)
        base_pipe, s_train = _fit_predict_pipeline(model_obj, pre, X_train, y_train, X_train)

    # transformed matrices for surrogate
    Z_train = base_pipe.named_steps["pre"].transform(X_train)
    # feature names
    try:
        feat_names = base_pipe.named_steps["pre"].get_feature_names_out().tolist()
    except Exception:
        # fallback: label features by index
        feat_names = [f"f_{i}" for i in range(Z_train.shape[1])]

    # keep surrogate simple & robust
    min_leaf = max(200, int(0.01 * len(train)))
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=min_leaf, random_state=42)
    tree.fit(Z_train, np.asarray(s_train, dtype=float))

    rule_specs = _extract_rule_specs_from_tree(tree, feat_names, cat_cols)

    # imputation maps for applying rules on raw df
    num_med, cat_mode = _impute_maps(train, num_cols, cat_cols)

    evaluated = []
    for rr in rule_specs[:40]:
        spec = rr["spec"]
        tr_f = _apply_rule_spec(train, spec, num_med, cat_mode)
        va_f = _apply_rule_spec(val, spec, num_med, cat_mode)
        te_f = _apply_rule_spec(test, spec, num_med, cat_mode)

        if len(tr_f) < 300 or len(va_f) < 120 or len(te_f) < 120:
            continue

        tr_roi = _roi(tr_f, pl_col, side, odds_col)
        va_roi = _roi(va_f, pl_col, side, odds_col)
        te_roi = _roi(te_f, pl_col, side, odds_col)

        gap_val = float(tr_roi.get("roi", 0.0)) - float(va_roi.get("roi", 0.0))
        te_games = _game_level(te_f, pl_col)

        # rank by VAL only (no tuning on TEST)
        score = float(va_roi.get("roi", 0.0)) - 0.25 * max(gap_val, 0.0)

        evaluated.append(
            {
                "score_val": score,
                "spec": spec,
                "train": tr_roi,
                "val": va_roi,
                "test": te_roi,
                "test_game_level": te_games,
                "gap_train_minus_val": gap_val,
                "samples": {
                    "train_rows": int(len(tr_f)),
                    "val_rows": int(len(va_f)),
                    "test_rows": int(len(te_f)),
                    "train_unique_ids": int(tr_f["ID"].nunique()) if "ID" in tr_f.columns else None,
                    "val_unique_ids": int(va_f["ID"].nunique()) if "ID" in va_f.columns else None,
                    "test_unique_ids": int(te_f["ID"].nunique()) if "ID" in te_f.columns else None,
                },
            }
        )

    evaluated.sort(key=lambda x: x["score_val"], reverse=True)

    return {
        "best_base_model": {"kind": best_kind, "model": best_model_name},
        "surrogate": {"type": "DecisionTreeRegressor", "max_depth": 4, "min_samples_leaf": min_leaf},
        "top_distilled_rules": evaluated[:10],
    }


# ============================================================
# Classic strategy_search (your earlier worker)
# ============================================================
def _rule_search(df, pl_col: str, side: str, odds_col: str, split_ratio: float = 0.7) -> Dict[str, Any]:
    import pandas as pd
    import numpy as np

    numeric_cols = [
        "DIFF",
        "% DRIFT",
        "ACTUAL ODDS",
        odds_col,
        "H XG VS A XG S",
        "H XG VS A XG 6",
        "Points Diff",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    if pl_col not in df.columns:
        return {"error": f"Missing {pl_col}"}
    if odds_col not in df.columns:
        return {"error": f"Missing odds column {odds_col}"}

    d0 = df[df[pl_col].notna()].copy()
    d0[pl_col] = pd.to_numeric(d0[pl_col], errors="coerce")
    d0 = d0[d0[pl_col].notna()].copy()

    for c in available:
        d0[c] = pd.to_numeric(d0[c], errors="coerce")

    train, test = _time_split(d0, split_ratio)

    def qgrid(s: pd.Series):
        s2 = s.dropna()
        if len(s2) < 500:
            return []
        qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        vals = [float(s2.quantile(q)) for q in qs]
        return [(vals[i], vals[i + 1]) for i in range(len(vals) - 1)]

    grids = {c: qgrid(train[c]) for c in available}
    cols = [c for c in available if grids.get(c)]
    if len(cols) < 2:
        return {"error": "Not enough numeric columns with stable quantile grids to search."}

    best = []
    max_rules = 250
    tried = 0

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            for (a1, b1) in grids[c1]:
                for (a2, b2) in grids[c2]:
                    tried += 1
                    if tried > max_rules:
                        break

                    tr = train[train[c1].between(a1, b1) & train[c2].between(a2, b2)]
                    te = test[test[c1].between(a1, b1) & test[c2].between(a2, b2)]

                    if len(tr) < 300 or len(te) < 120:
                        continue

                    tr_roi = _roi(tr, pl_col, side, odds_col)
                    te_roi = _roi(te, pl_col, side, odds_col)

                    gap = float(tr_roi.get("roi", 0.0)) - float(te_roi.get("roi", 0.0))
                    te_games = _game_level(te, pl_col)
                    score = float(te_roi.get("roi", 0.0)) - 0.25 * max(gap, 0.0)

                    best.append(
                        {
                            "score": score,
                            "rule": [
                                {"col": c1, "min": a1, "max": b1},
                                {"col": c2, "min": a2, "max": b2},
                            ],
                            "train": tr_roi,
                            "test": te_roi,
                            "test_game_level": te_games,
                            "gap_train_minus_test": gap,
                            "samples": {"train_rows": int(len(tr)), "test_rows": int(len(te))},
                        }
                    )

                if tried > max_rules:
                    break
            if tried > max_rules:
                break
        if tried > max_rules:
            break

    best.sort(key=lambda x: x["score"], reverse=True)
    return {
        "pl_column": pl_col,
        "side": side,
        "odds_col": odds_col,
        "searched_rules": int(tried),
        "top_rules": best[:10],
    }


def _pick_market(df, split_ratio: float = 0.7) -> Dict[str, Any]:
    mp = _mapping()
    candidates = []
    for pl_col, (side, odds_col) in mp.items():
        if pl_col not in df.columns:
            continue
        filtered = df[df[pl_col].notna()].copy()
        train, test = _time_split(filtered, split_ratio)
        test_stats = _roi(test, pl_col, side, odds_col)
        candidates.append((pl_col, side, odds_col, float(test_stats.get("roi", -999)), int(test_stats.get("rows", 0))))
    candidates.sort(key=lambda x: (x[3], x[4]), reverse=True)
    if not candidates:
        return {"error": "No PL columns found to choose from."}
    pl_col, side, odds_col, test_roi, test_rows = candidates[0]
    return {"pl_column": pl_col, "side": side, "odds_col": odds_col, "test_roi": float(test_roi), "test_rows": int(test_rows)}


def _compute_task_strategy_search(sb, params: Dict[str, Any]) -> Dict[str, Any]:
    split_ratio = float(params.get("time_split_ratio", 0.7))

    storage_bucket = params.get("storage_bucket")
    storage_path = params.get("storage_path")
    if not storage_bucket or not storage_path:
        return {"error": "strategy_search requires params.storage_bucket and params.storage_path"}

    df = _load_csv_from_supabase_storage(sb, storage_bucket, storage_path)

    chosen = params.get("pl_column") or params.get("target_pl_column")
    mp = _mapping()

    if not chosen:
        picked = _pick_market(df, split_ratio)
        if "error" in picked:
            return picked
        chosen = picked["pl_column"]
        side = picked["side"]
        odds_col = picked["odds_col"]
        pick_meta = picked
    else:
        chosen = str(chosen).strip()
        if chosen in mp:
            side, odds_col = mp[chosen]
        else:
            side = (params.get("side") or "back").strip().lower()
            odds_col = params.get("odds_col")
        if not odds_col:
            return {"error": f"No mapping for {chosen}. Provide params.odds_col and params.side."}
        pick_meta = {"pl_column": chosen, "side": side, "odds_col": odds_col}

    out = _rule_search(df, chosen, side, odds_col, split_ratio)
    return {"picked": pick_meta, "search": out}


# ============================================================
# ML lab (general: works for any PL column, includes distillation w/ categoricals)
# ============================================================
def _strategy_from_scores(df_split, scores, pl_col: str, side: str, odds_col: str, top_frac: float) -> Dict[str, Any]:
    import numpy as np

    d = df_split.copy()
    d["_score"] = np.asarray(scores, dtype=float)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["_score"])

    if len(d) == 0:
        return {"error": "No valid scores."}

    top_frac = max(0.01, min(float(top_frac), 0.5))
    k = max(1, int(round(len(d) * top_frac)))

    picked = d.nlargest(k, "_score")
    roi = _roi(picked, pl_col, side, odds_col)
    games = _game_level(picked, pl_col)

    return {
        "top_frac": top_frac,
        "picked_rows": int(len(picked)),
        "picked_unique_ids": int(picked["ID"].nunique()) if "ID" in picked.columns else None,
        "roi": roi,
        "game_level": games,
    }


def _compute_task_pl_lab(sb, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    ML lab that can target ANY PL column (BTTS PL, BO 2.5 PL, etc),
    uses time splits, runs multiple models, and distills a final explicit rule set
    INCLUDING categoricals.
    """
    import pandas as pd

    storage_bucket = params.get("storage_bucket")
    storage_path = params.get("storage_path")
    if not storage_bucket or not storage_path:
        return {"error": "pl_lab requires params.storage_bucket and params.storage_path"}

    pl_col = str(params.get("pl_column") or "BTTS PL").strip()
    mp = _mapping()
    if pl_col in mp:
        side, odds_col = mp[pl_col]
    else:
        side = (params.get("side") or "back").strip().lower()
        odds_col = params.get("odds_col")

    if not odds_col:
        return {"error": f"Cannot resolve odds_col for {pl_col}. Provide params.odds_col or add to mapping."}

    ignored_columns = params.get("ignored_columns") or []
    outcome_columns = params.get("outcome_columns") or []

    duration_minutes = max(5, min(int(params.get("duration_minutes") or 60), 360))
    deadline = time.time() + duration_minutes * 60

    top_fracs = params.get("top_fracs") or [0.05, 0.1, 0.2]
    top_n = max(3, min(int(params.get("top_n") or 12), 25))

    df = _load_csv_from_supabase_storage(sb, storage_bucket, storage_path)
    if pl_col not in df.columns:
        return {"error": f"Missing PL column in CSV: {pl_col}", "pl_like": [c for c in df.columns if "PL" in str(c)]}

    d0 = df[df[pl_col].notna()].copy()
    d0[pl_col] = pd.to_numeric(d0[pl_col], errors="coerce")
    d0 = d0[d0[pl_col].notna()].copy()
    if len(d0) < 3000:
        return {"error": "Too few rows with target PL to run ML lab reliably.", "rows_target": int(len(d0))}

    d0 = _time_sort(d0)

    feature_cols = _safe_feature_columns(d0, pl_col, ignored_columns, outcome_columns)
    train, val, test = _time_split_3way(d0, 0.6, 0.2)

    # Baselines (no filtering)
    baseline_train = _roi(train, pl_col, side, odds_col)
    baseline_val = _roi(val, pl_col, side, odds_col)
    baseline_test = _roi(test, pl_col, side, odds_col)

    pre, num_cols, cat_cols = _build_preprocessor(train, feature_cols)

    X_train = train[feature_cols]
    X_val = val[feature_cols]
    X_test = test[feature_cols]

    y_train_reg = train[pl_col].astype(float).values
    y_val_reg = val[pl_col].astype(float).values
    y_test_reg = test[pl_col].astype(float).values

    y_train_clf = (y_train_reg > 0).astype(int)
    y_val_clf = (y_val_reg > 0).astype(int)
    y_test_clf = (y_test_reg > 0).astype(int)

    regressors = _make_models("regression")
    classifiers = _make_models("classification")

    leaderboard = {"regression": [], "classification": []}
    candidates: List[Tuple[float, Dict[str, Any]]] = []

    # regression candidates
    for name, model in regressors.items():
        if time.time() > deadline:
            break
        _, pred_val = _fit_predict_pipeline(model, pre, X_train, y_train_reg, X_val)
        _, pred_test = _fit_predict_pipeline(model, pre, X_train, y_train_reg, X_test)

        val_strats = [_strategy_from_scores(val, pred_val, pl_col, side, odds_col, f) for f in top_fracs]
        test_strats = [_strategy_from_scores(test, pred_test, pl_col, side, odds_col, f) for f in top_fracs]
        entry = {"model": name, "task": "regression", "val_strategies": val_strats, "test_strategies": test_strats}
        leaderboard["regression"].append(entry)

        pick = next((s for s in val_strats if abs(s.get("top_frac", 0) - 0.1) < 1e-9), val_strats[0])
        val_roi = float((pick.get("roi") or {}).get("roi", -999))
        val_dd = float((pick.get("game_level") or {}).get("max_dd", 0.0))
        score = val_roi + 0.01 * val_dd
        candidates.append((score, {"kind": "ml_reg", "model": name}))

    # classification candidates
    for name, model in classifiers.items():
        if time.time() > deadline:
            break
        _, p_val = _fit_predict_pipeline(model, pre, X_train, y_train_clf, X_val)
        _, p_test = _fit_predict_pipeline(model, pre, X_train, y_train_clf, X_test)

        val_strats = [_strategy_from_scores(val, p_val, pl_col, side, odds_col, f) for f in top_fracs]
        test_strats = [_strategy_from_scores(test, p_test, pl_col, side, odds_col, f) for f in top_fracs]
        entry = {"model": name, "task": "classification_pos_pl", "val_strategies": val_strats, "test_strategies": test_strats}
        leaderboard["classification"].append(entry)

        pick = next((s for s in val_strats if abs(s.get("top_frac", 0) - 0.1) < 1e-9), val_strats[0])
        val_roi = float((pick.get("roi") or {}).get("roi", -999))
        val_dd = float((pick.get("game_level") or {}).get("max_dd", 0.0))
        score = val_roi + 0.01 * val_dd
        candidates.append((score, {"kind": "ml_clf", "model": name}))

    candidates.sort(key=lambda x: float(x[0]), reverse=True)
    best = [c for _, c in candidates[:top_n]]

    # distill best -> explicit strategy rules INCLUDING categoricals
    distilled = None
    try:
        if best:
            best_one = best[0]
            best_kind = best_one.get("kind")
            best_model_name = best_one.get("model")
            model_obj = regressors.get(best_model_name) if best_kind == "ml_reg" else classifiers.get(best_model_name)
            if model_obj is not None:
                distilled = _distill_best_model_to_rules(
                    best_kind=best_kind,
                    best_model_name=best_model_name,
                    model_obj=model_obj,
                    pre=pre,
                    train=train,
                    val=val,
                    test=test,
                    feature_cols=feature_cols,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    pl_col=pl_col,
                    side=side,
                    odds_col=odds_col,
                )
    except Exception as e:
        distilled = {"error": f"distillation failed: {e}"}

    return {
        "picked": {"pl_column": pl_col, "side": side, "odds_col": odds_col},
        "sheet_enforcement": {"ignored_columns": ignored_columns, "outcome_columns": outcome_columns},
        "dataset": {"rows_total": int(len(df)), "rows_target": int(len(d0))},
        "splits": {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_unique_ids": int(train["ID"].nunique()) if "ID" in train.columns else None,
            "val_unique_ids": int(val["ID"].nunique()) if "ID" in val.columns else None,
            "test_unique_ids": int(test["ID"].nunique()) if "ID" in test.columns else None,
        },
        "features": {"n_features_total": int(len(feature_cols)), "n_numeric": int(len(num_cols)), "n_categorical": int(len(cat_cols))},
        "baseline": {"train": baseline_train, "val": baseline_val, "test": baseline_test},
        "leaderboard": leaderboard,
        "best_candidates": best,
        "distilled": distilled,
        "ts": _now_iso(),
        "duration_minutes": duration_minutes,
    }


# ============================================================
# Job processing
# ============================================================
def _process_one(sb) -> Dict[str, Any]:
    job_rows = (
        sb.table("jobs")
        .select("*")
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
        .data
    )
    if not job_rows:
        return {"status": "idle", "message": "No queued jobs."}

    job = job_rows[0]
    job_id = job["job_id"]

    sb.table("jobs").update({"status": "running", "updated_at": _now_iso()}).eq("job_id", job_id).execute()

    try:
        task_type = job.get("task_type")
        params = job.get("params") or {}
        if not isinstance(params, dict):
            raise RuntimeError("params must be a JSON object")

        if task_type == "ping":
            result = {"job_id": job_id, "task_type": task_type, "params": params, "computed_at": _now_iso(), "message": "OK"}
        elif task_type == "strategy_search":
            result = {"job_id": job_id, "task_type": task_type, "params": params, "computed_at": _now_iso(), "result": _compute_task_strategy_search(sb, params)}
        elif task_type in ("btts_lab", "pl_lab"):
            # keep backward compatible name: btts_lab -> pl_lab
            result = {"job_id": job_id, "task_type": task_type, "params": params, "computed_at": _now_iso(), "result": _compute_task_pl_lab(sb, params)}
        else:
            result = {"job_id": job_id, "task_type": task_type, "params": params, "computed_at": _now_iso(), "error": f"Unknown task_type: {task_type}"}

        bucket = (params.get("_results_bucket") or "football-results").strip()
        path = f"results/{job_id}.json"
        payload = json.dumps(result, indent=2).encode("utf-8")

        sb.storage.from_(bucket).upload(
            path=path,
            file=payload,
            file_options={"content-type": "application/json", "upsert": "true"},
        )

        sb.table("jobs").update({"status": "done", "result_path": path, "updated_at": _now_iso(), "error": None}).eq("job_id", job_id).execute()
        return {"status": "done", "job_id": job_id, "result_path": path, "bucket": bucket}

    except Exception as e:
        sb.table("jobs").update({"status": "error", "error": str(e), "updated_at": _now_iso()}).eq("job_id", job_id).execute()
        return {"status": "error", "job_id": job_id, "error": str(e)}


# ============================================================
# Modal functions
# ============================================================
@app.function(image=image, secrets=[SUPABASE_SECRET], timeout=6 * 60 * 60)
def run_batch(max_jobs: int = 10) -> Dict[str, Any]:
    import os
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    sb = create_client(url, key)

    max_jobs = max(1, min(int(max_jobs or 10), 50))
    results = []
    for _ in range(max_jobs):
        out = _process_one(sb)
        results.append(out)
        if out.get("status") == "idle":
            break
    return {"status": "ok", "processed": len(results), "results": results, "ts": _now_iso()}


@app.function(image=image, secrets=[SUPABASE_SECRET], timeout=60 * 60, schedule=modal.Period(minutes=1))
def poll_and_run():
    return run_batch.remote(max_jobs=10)


@app.local_entrypoint()
def main():
    print(run_batch.remote(max_jobs=5))
