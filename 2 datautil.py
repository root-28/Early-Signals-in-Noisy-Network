"""Loading, cleaning, and splitting the dataset."""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _pick_target(df, preferred):
    if preferred in df.columns:
        return preferred

    candidates = ["label", "class", "target", "attack", "category"]
    for c in candidates:
        if c in df.columns:
            print(f"[data] '{preferred}' not found, using '{c}'")
            return c

    raise ValueError(
        f"Couldn't find a target column. Tried: {preferred}, {candidates}"
    )


def load_and_clean(path, target_column="label"):
    """Read CSV, drop junk rows, return (X, y, target_name, label_encoder)."""

    print(f"[data] reading {path}")
    df = pd.read_csv(path)
    print(f"[data] raw shape: {df.shape}")

    target = _pick_target(df, target_column)

    # --- dedupe ----------------------------------------------------------
    n0 = len(df)
    df = df.drop_duplicates()
    print(f"[data] dropped {n0 - len(df)} duplicate rows")

    # --- inf -> nan -> drop ---------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    n1 = len(df)
    df = df.dropna()
    print(f"[data] dropped {n1 - len(df)} rows with NaN/inf")

    X = df.drop(columns=[target])
    y = df[target]

    # --- encode any leftover object columns ------------------------------
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        print(f"[data] label-encoding object columns: {obj_cols}")
        for col in obj_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.astype(np.float32)

    # --- encode target ---------------------------------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    print(f"[data] {len(le.classes_)} classes: {list(le.classes_)}")

    return X, y_enc, target, le


def split_and_scale(X, y, test_size=0.20, seed=42):
    """Stratified split + StandardScaler fitted on train only."""

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    print(f"[data] train: {X_tr.shape}  test: {X_te.shape}")
    return X_tr, X_te, y_tr, y_te, scaler


def to_lstm_shape(X):
    """(n, features) -> (n, features, 1). Each feature is a timestep."""
    return X.reshape(X.shape[0], X.shape[1], 1)