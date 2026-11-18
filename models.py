"""
models.py
---------
Model definitions and the early-stopping training loop used for both the
Baseline ANN and the Neuro-Fuzzy ANN (see report, Chapter 4, Section 4.5).
Both models share an identical architecture and optimiser configuration;
only their *input feature space* differs. A plain Logistic Regression
baseline is also provided.
"""

from __future__ import annotations
import copy
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
HIDDEN_LAYERS = (16, 8)
LEARNING_RATE_INIT = 0.01
ALPHA = 1e-3
MAX_EPOCHS = 150
PATIENCE = 15


def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam",
        alpha=ALPHA, max_iter=1, warm_start=True, random_state=RANDOM_STATE,
        learning_rate_init=LEARNING_RATE_INIT,
    )


def train_with_early_stopping(X_tr, y_tr, X_val, y_val, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Epoch-by-epoch training with manual early stopping on validation loss
    (binary cross-entropy), returning the best checkpoint -- exactly the
    protocol documented in Section 4.5 of the report."""
    model = make_mlp()
    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(max_epochs):
        model.fit(X_tr, y_tr)
        p_val = model.predict_proba(X_val)[:, 1]
        p_tr = model.predict_proba(X_tr)[:, 1]
        eps = 1e-9
        val_loss = -np.mean(y_val * np.log(p_val + eps) + (1 - y_val) * np.log(1 - p_val + eps))
        history["train_loss"].append(model.loss_)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(accuracy_score(y_tr, p_tr > 0.5))
        history["val_acc"].append(accuracy_score(y_val, p_val > 0.5))
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model)
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    return best_state, best_epoch, history


def make_logreg():
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
