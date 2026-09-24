"""A lightweight stochastic search for LSTM hyperparameters.

This is *inspired by* AVOA but not a faithful implementation. It keeps a
population, evaluates each candidate, and refreshes the population around
the best-so-far with some jitter. Good enough for a coarse hyperparameter
sweep; if you need the real algorithm (with vulture starvation phases),
swap this out.
"""

import gc
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from models import build_lstm


def _sample():
    return {
        "units": random.choice([32, 64, 128]),
        "dropout": random.uniform(0.10, 0.50),
        "lr": 10 ** random.uniform(-4, -2),
    }


def _jitter(best):
    """New candidate biased toward the current best."""
    return {
        "units": best["units"],
        "dropout": float(np.clip(best["dropout"] + random.uniform(-0.05, 0.05), 0.10, 0.50)),
        "lr": float(np.clip(best["lr"] * random.uniform(0.7, 1.3), 1e-4, 1e-2)),
    }


def search(X_train, y_train, num_classes, input_shape,
           pop=4, iters=5, fitness_epochs=5, batch=64,
           val_split=0.10, seed=42):
    """Run the search. Returns dict of best hyperparameters."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Fixed validation slice so fitness is comparable across candidates.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_split,
        random_state=seed,
        stratify=y_train,
    )

    population = [_sample() for _ in range(pop)]
    best, best_loss = None, float("inf")

    print("\n[avoa] starting search")
    t0 = time.time()

    for it in range(iters):
        print(f"[avoa] iteration {it + 1}/{iters}")
        scored = []

        for i, cand in enumerate(population):
            tf.keras.backend.clear_session()

            model = build_lstm(input_shape, num_classes,
                               units=cand["units"],
                               dropout=cand["dropout"],
                               lr=cand["lr"])

            hist = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=fitness_epochs,
                batch_size=batch,
                verbose=0,
                callbacks=[EarlyStopping(monitor="val_loss",
                                         patience=2,
                                         restore_best_weights=True)],
            )

            loss = float(min(hist.history["val_loss"]))
            scored.append((loss, cand))

            print(f"[avoa]   cand {i + 1}: "
                  f"loss={loss:.4f}  units={cand['units']}  "
                  f"dropout={cand['dropout']:.3f}  lr={cand['lr']:.5f}")

            if loss < best_loss:
                best_loss, best = loss, cand.copy()

            del model, hist
            gc.collect()

        # Refresh: 60% around best, 40% fresh.
        new_pop = []
        for _ in range(pop):
            new_pop.append(_jitter(best) if random.random() < 0.6 else _sample())
        population = new_pop

    print(f"[avoa] done in {time.time() - t0:.1f}s")
    print(f"[avoa] best: units={best['units']} "
          f"dropout={best['dropout']:.3f} lr={best['lr']:.5f} "
          f"(val_loss={best_loss:.4f})")

    return best, best_loss