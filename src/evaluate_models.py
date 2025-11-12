#Evaluate models on performance + efficiency (time, memory)
import os, time, tracemalloc, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


#Config

OUT_DIR = "./out"
USE_SMOTE = True
SUBSAMPLE_TRAIN_FRAC = 0.25
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)


#Load preprocessed arrays

X = np.load(os.path.join(OUT_DIR, "X.npy"))
y = np.load(os.path.join(OUT_DIR, "y.npy"))


#Train/test split (stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=RANDOM_STATE, stratify=y
)


#SMOTE on TRAIN only (optional)

if USE_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[info] SMOTE applied on train: {len(y_train)} samples")
    except Exception as e:
        print(f"[warn] SMOTE not available ({e}); continuing without it.")


#Optional subsampling for speed

if SUBSAMPLE_TRAIN_FRAC is not None and 0 < SUBSAMPLE_TRAIN_FRAC < 1.0:
    rs = np.random.RandomState(RANDOM_STATE)
    idx = rs.choice(len(X_train), size=int(len(X_train)*SUBSAMPLE_TRAIN_FRAC), replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]
    print(f"[info] Subsampled train to {len(y_train)} samples for speed.")


#LightGBM (optional)

LGBM_AVAILABLE = True
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    LGBM_AVAILABLE = False
    print(f"[warn] LightGBM not available ({e}); skipping LGBM. Install with: pip install lightgbm")


#Efficiency evaluation helper

def evaluate_model_efficiency(model, X_train, X_test, y_train, y_test, name="model"):
    # Train time + peak memory
    tracemalloc.start()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024*1024)

    #Prediction time
    t1 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t1
    throughput = len(y_test) / pred_time if pred_time > 0 else float('inf')

    #Metrics
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "train_time_sec": round(train_time, 2),
        "pred_time_sec": round(pred_time, 4),
        "peak_mem_mb": round(peak_mb, 2),
        "pred_throughput_sps": round(throughput, 2)  # samples per second
    }
    return metrics


#Models to evaluate

models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, n_jobs=-1)),
    ("RandomForest(n=200)", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ("GradientBoosting",   GradientBoostingClassifier(random_state=RANDOM_STATE)),
]
if LGBM_AVAILABLE:
    models.append(("LightGBM", LGBMClassifier(
        n_estimators=300, learning_rate=0.1, num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1
    )))


#Run & collect

results = []
for name, mdl in models:
    print(f"\n=== Running {name} ===")
    res = evaluate_model_efficiency(mdl, X_train, X_test, y_train, y_test, name=name)
    results.append(res)
    print(res)

res_df = pd.DataFrame(results)
res_csv = os.path.join(OUT_DIR, "model_efficiency_comparison.csv")
res_df.to_csv(res_csv, index=False)
print(f"\n[info] Saved results to {res_csv}")


#Simple charts (saved as PNG)
import matplotlib.pyplot as plt

#Performance: AUC and F1
fig1 = plt.figure(figsize=(8, 5))
ax = plt.gca()
res_df.plot(x="model", y=["auc", "f1"], kind="bar", ax=ax)
plt.title("Model Performance (AUC & F1)")
plt.ylabel("Score")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
perf_png = os.path.join(OUT_DIR, "perf_auc_f1.png")
plt.savefig(perf_png, dpi=150)
plt.close(fig1)

#Efficiency: time & memory
fig2 = plt.figure(figsize=(8, 5))
ax = plt.gca()
res_df.plot(x="model", y=["train_time_sec", "pred_time_sec", "peak_mem_mb"], kind="bar", ax=ax)
plt.title("Model Efficiency (Train/Pred Time & Peak RAM)")
plt.ylabel("Seconds / MB")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
eff_png = os.path.join(OUT_DIR, "efficiency_time_memory.png")
plt.savefig(eff_png, dpi=150)
plt.close(fig2)

print(f"[info] Saved charts:\n  {perf_png}\n  {eff_png}")
