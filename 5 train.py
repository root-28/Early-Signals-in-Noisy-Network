"""End-to-end training script.

Run:  python train.py
"""

import os
import time

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

import config
from data_utils import load_and_clean, split_and_scale, to_lstm_shape
from avoa import search as avoa_search
from models import build_lstm, make_feature_extractor


def _metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _print_metrics(tag, m):
    print(f"\n[{tag}]")
    print(f"  accuracy : {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}")
    print(f"  recall   : {m['recall']:.4f}")
    print(f"  f1 (w)   : {m['f1']:.4f}")
    print(f"  f1 (mac) : {m['f1_macro']:.4f}")


def main():
    # reproducibility ------------------------------------------------------
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

    print(f"[env] tensorflow {tf.__version__}")
    print(f"[env] gpus: {tf.config.list_physical_devices('GPU')}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # data ----------------------------------------------------------------
    X, y, target, label_enc = load_and_clean(
        config.DATA_PATH, config.TARGET_COLUMN
    )
    num_classes = len(label_enc.classes_)

    X_tr, X_te, y_tr, y_te, _ = split_and_scale(
        X, y, test_size=config.TEST_SIZE, seed=config.SEED
    )

    X_tr_l = to_lstm_shape(X_tr)
    X_te_l = to_lstm_shape(X_te)
    input_shape = X_tr_l.shape[1:]

    # AVOA ----------------------------------------------------------------
    best, best_loss = avoa_search(
        X_tr_l, y_tr,
        num_classes=num_classes,
        input_shape=input_shape,
        pop=config.AVOA_POP,
        iters=config.AVOA_ITERS,
        fitness_epochs=config.AVOA_FITNESS_EPOCHS,
        batch=64,
        val_split=config.AVOA_VAL_SPLIT,
        seed=config.SEED,
    )

    # final LSTM ----------------------------------------------------------
    print("\n[train] fitting final LSTM")
    lstm = build_lstm(
        input_shape, num_classes,
        units=best["units"],
        dropout=best["dropout"],
        lr=best["lr"],
        feature_dim=config.FEATURE_DIM,
    )
    lstm.summary()

    t0 = time.time()
    hist = lstm.fit(
        X_tr_l, y_tr,
        validation_split=config.LSTM_VAL_SPLIT,
        epochs=config.LSTM_MAX_EPOCHS,
        batch_size=config.LSTM_BATCH,
        verbose=2,
        callbacks=[EarlyStopping(monitor="val_loss",
                                 patience=config.LSTM_PATIENCE,
                                 restore_best_weights=True)],
    )
    print(f"[train] LSTM done in {time.time() - t0:.1f}s")

    # LSTM-only baseline --------------------------------------------------
    prob = lstm.predict(X_te_l, batch_size=512, verbose=0)
    lstm_pred = np.argmax(prob, axis=1)
    lstm_m = _metrics(y_te, lstm_pred)
    _print_metrics("lstm only", lstm_m)

    # extract features ----------------------------------------------------
    print("\n[train] extracting deep features")
    extractor = make_feature_extractor(lstm)
    F_tr = extractor.predict(X_tr_l, batch_size=512, verbose=0)
    F_te = extractor.predict(X_te_l, batch_size=512, verbose=0)
    print(f"[train] feature shapes: train={F_tr.shape}  test={F_te.shape}")

    # XGBoost -------------------------------------------------------------
    print("\n[train] fitting XGBoost on deep features")
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=config.SEED,
        **config.XGB_PARAMS,
    )
    t0 = time.time()
    xgb.fit(F_tr, y_tr)
    print(f"[train] XGBoost done in {time.time() - t0:.1f}s")

    xgb_pred = xgb.predict(F_te)
    xgb_m = _metrics(y_te, xgb_pred)
    _print_metrics("lstm + xgboost", xgb_m)

    # Bagging -------------------------------------------------------------
    print("\n[train] fitting bagging ensemble")
    base = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=config.SEED,
        n_jobs=1,
        **{k: v for k, v in config.XGB_PARAMS.items() if k != "n_jobs"},
    )
    bag = BaggingClassifier(
        estimator=base,
        n_estimators=config.BAGGING_N_ESTIMATORS,
        max_samples=config.BAGGING_MAX_SAMPLES,
        bootstrap=True,
        n_jobs=-1,
        random_state=config.SEED,
    )
    t0 = time.time()
    bag.fit(F_tr, y_tr)
    print(f"[train] bagging done in {time.time() - t0:.1f}s")

    final_pred = bag.predict(F_te)
    final_m = _metrics(y_te, final_pred)
    _print_metrics("meta-hybrid (final)", final_m)

    # report --------------------------------------------------------------
    print("\n[report] classification report (final)")
    print(classification_report(
        y_te, final_pred,
        target_names=label_enc.classes_,
        zero_division=0,
    ))

    # save ----------------------------------------------------------------
    rows = [
        {"model": "LSTM + AVOA", **lstm_m},
        {"model": "LSTM + AVOA + XGBoost", **xgb_m},
        {"model": "Meta-Hybrid (+ Bagging)", **final_m},
    ]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(config.OUTPUT_DIR, "metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[save] wrote {out_csv}")

    lstm.save(os.path.join(config.OUTPUT_DIR, "lstm.keras"))
    print(f"[save] wrote {config.OUTPUT_DIR}/lstm.keras")

    preds = pd.DataFrame({"y_true": y_te, "y_pred": final_pred})
    preds.to_csv(os.path.join(config.OUTPUT_DIR, "predictions.csv"), index=False)
    print(f"[save] wrote {config.OUTPUT_DIR}/predictions.csv")


if __name__ == "__main__":
    main()