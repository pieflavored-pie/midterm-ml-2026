# dataset from https://www.kaggle.com/datasets/nitikachandel95/online-learning-engagement-and-performance-oulad
# more info: https://analyse.kmi.open.ac.uk/open-dataset


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# from sklearn.linear_model import LogisticRegression # only for fast prototypes

# =============================================================
# PART 1 — PREPROCESSING (teammate's code)
# =============================================================

df = pd.read_csv("online_education_dataset.csv")


# just remove those who have withdrawn entirely
df = df[df['final_result'] != 'Withdrawn'].copy()

df = df.drop(["id_student", "gender", "region", "engagement_level",
              "performance_level", "risk_level", "dropout_flag", "final_result"], axis=1)
feature_names = [
    "highest_education_numeric", "studied_credits", "imd_band_numeric",
    "imd_missing", "total_clicks", "avg_score",
]
"""
processing numeric features that have nulls: total_clicks, avg_score:
fill with 0 since those students did not participate in the courses
"""
fillna_val = {"total_clicks": 0, "avg_score": 0}
df.fillna(value=fillna_val, inplace=True)
#df["total_clicks"] = np.log1p(df["total_clicks"]) # log transform (returns ln(1+x) for np.log1p(x))
#df["studied_credits"] = np.log1p(df["studied_credits"]) # log transform

"""
processing categorical features: highest_education, imd_band
1. highest_education: ordinal encoding (education level from low to high)
2. imd_band: students from North Region and Ireland don't have IMD (probably the gov doesn't include those regions)
- create a column acting as an indicator: IMD missing or not
- then apply ordinal encoding, filling nulls with median, assuming average
"""
# 1. highest_education
edu_lvl_map = {
    "No Formal quals": 1, "Lower Than A Level": 2, "A Level or Equivalent": 3,
    "HE Qualification": 4, "Post Graduate Qualification": 5
}
df["highest_education_numeric"] = df["highest_education"].map(edu_lvl_map)

# 2. imd_band
# null value gets 1, everything else gets 0
df["imd_missing"] = df["imd_band"].isnull().astype(int)

# ordinal encoding + filling nulls
imd_map = {
    '0-10%': 1, '10-20': 2, '20-30%': 3, '30-40%': 4, '40-50%': 5,
    '50-60%': 6, '60-70%': 7, '70-80%': 8, '80-90%': 9, '90-100%': 10
}
df["imd_band_numeric"] = df["imd_band"].map(imd_map)
df["imd_band_numeric"] = df["imd_band_numeric"].fillna(df["imd_band_numeric"].median())


"""
splitting dataset:
1. convert to numpy array
2. split into training, validation, and test (60/20/20)
"""
# X: matrix containing all features
X = df[["highest_education_numeric", "studied_credits", "imd_band_numeric",
        "imd_missing", "total_clicks", "avg_score"]].to_numpy()

# Y: column vector containing targets
Y = df[["pass_flag"]].to_numpy()

# 1. split into train and test
rand_state = 43
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=rand_state, stratify=Y)
# 2. split X_train into X_train and X_val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=rand_state, stratify=y_train)
# use X_train to train, X_val to validate, X_test to test
print(f"\nSplits: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
# =============================================================
# PART 1 — LOGISTIC REGRESSION ALGORITHM (teammate's code)
# =============================================================
print("\n" + "=" * 70)
print("LOGISTIC REGRESSION ALGORITHM")
print("=" * 70)


def sigmoid(z):
    """Numerically stable sigmoid: sigmoid(z) = 1 / (1 + e^{-z})"""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost(X, y, w, b, reg_type="none", lam=0.0):
    """
    J(w,b) = -(1/m) * sum[y*log(h) + (1-y)*log(1-h)]  +  regularization

    reg_type:
      "none" -> no penalty
      "l2"   -> + lambda/(2m) * ||w||^2   (Ridge)
      "l1"   -> + lambda/m * ||w||_1      (Lasso)
    """
    m = len(y)
    h = sigmoid(X @ w + b)
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)

    bce = -(1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))

    if reg_type == "l2":
        bce += (lam / (2 * m)) * np.sum(w ** 2)
    elif reg_type == "l1":
        bce += (lam / m) * np.sum(np.abs(w))
    return bce


def compute_gradients(X, y, w, b, reg_type="none", lam=0.0):
    """
    dw = (1/m) * X^T * (h - y)  +  regularization gradient
    db = (1/m) * sum(h - y)

    Ridge gradient: + (lambda/m) * w
    Lasso gradient: + (lambda/m) * sign(w)
    Note: bias b is NEVER regularized
    """
    m = len(y)
    h = sigmoid(X @ w + b)
    error = h - y

    dw = (1 / m) * (X.T @ error)
    db = (1 / m) * np.sum(error)

    if reg_type == "l2":
        dw += (lam / m) * w
    elif reg_type == "l1":
        dw += (lam / m) * np.sign(w)
    return dw, db


def train_model(X_tr, y_tr, X_v, y_v, lr=0.5, epochs=1000,
                reg_type="none", lam=0.0, verbose=True):
    """
    Full gradient descent training loop.
    Returns: w, b, train_cost_history, val_cost_history
    """
    n_features = X_tr.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    train_costs, val_costs = [], []

    for ep in range(epochs):
        t_cost = compute_cost(X_tr, y_tr, w, b, reg_type, lam)
        train_costs.append(t_cost)
        v_cost = compute_cost(X_v, y_v, w, b, reg_type, lam)
        val_costs.append(v_cost)

        dw, db = compute_gradients(X_tr, y_tr, w, b, reg_type, lam)
        w -= lr * dw
        b -= lr * db

        if verbose and (ep % 200 == 0 or ep == epochs - 1):
            print(f"    Epoch {ep:>4d} | Train: {t_cost:.6f} | Val: {v_cost:.6f}")

    return w, b, train_costs, val_costs


def predict(X, w, b, threshold=0.5):
    probs = sigmoid(X @ w + b)
    return (probs >= threshold).astype(int), probs


#Part 2 :Train 3 models

y_train_1d = y_train.ravel()
y_val_1d   = y_val.ravel()
y_test_1d  = y_test.ravel()


mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8
X_train_s = (X_train - mean) / std
X_val_s   = (X_val   - mean) / std
X_test_s  = (X_test  - mean) / std


LR         = 0.05
EPOCHS_MAP = {
    "Zero":  20,  #create 3 models 20/500/2000 epochs
    "One":  500,
    "Two":   2000,
}
THRES = 0.65

models = {}

print("=" * 60)
print("Training 3 models...")
print("=" * 60)

for name, epochs in EPOCHS_MAP.items():
    print(f"\n── {name} (epochs={epochs}) ──")
    w, b, tr_costs, va_costs = train_model(
        X_train_s, y_train_1d,
        X_val_s, y_val_1d,
        lr=LR,
        epochs=epochs,
        reg_type="none",
        verbose=True,
    )
    models[name] = dict(w=w, b=b, train_costs=tr_costs, val_costs=va_costs,
                        epochs=epochs)

#plot 3 graph
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("3 Models: Zero / One / Two", fontsize=15, fontweight="bold")

colors = {"train": "#2196F3", "val": "#F44336"}

for col, (name, m) in enumerate(models.items()):
    ax = axes[col]
    ep = range(1, m["epochs"] + 1)
    ax.plot(ep, m["train_costs"], color=colors["train"], label="Train loss")
    ax.plot(ep, m["val_costs"],   color=colors["val"],   label="Val loss", linestyle="--")
    ax.set_title(f"{name}\n(epochs = {m['epochs']})", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.annotate(f"Train: {m['train_costs'][-1]:.4f}\nVal:   {m['val_costs'][-1]:.4f}",
                xy=(0.97, 0.75), xycoords="axes fraction",
                ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("3_models_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[Saved] 3_models_comparison.png")
print("\n" + "=" * 60)
print("Models ready. Pass `models`, `X_test_s`, `y_test_1d` to next parts.")
print("=" * 60)

# evaluation
def evaluate(y_true, y_pred, y_prob, name):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f_one_half = 1.25 * prec * rec / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0 # f0.5 favours precision
    cm = np.array([[tn, fp], [fn, tp]])

    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Specificity:  {spec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  F0.5 Score:  {f_one_half:.4f}")
    print(f"  Confusion Matrix:  TN={tn}  FP={fp}")
    print(f"                     FN={fn}  TP={tp}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm}
# 1. evaluate on training set
print("EVALUATION ON TRAINING SET")
results = {}

actual = []
predicted = []

for name, theta in models.items():
  y_pred, y_prob = predict(X_train_s, theta["w"], theta["b"], threshold=THRES)
  results[name] = evaluate(y_train_1d, y_pred, y_prob, name)


# 2. evaluate on test set
print("EVALUATION ON TEST SET")
results = {}
i = 0
for name, theta in models.items():
  y_pred, y_prob = predict(X_test_s, theta["w"], theta["b"], threshold=THRES)
  results[name] = evaluate(y_test_1d, y_pred, y_prob, name)
  actual.append(y_test_1d)
  predicted.append(y_pred)

for i in range(3):
  confus_mat = metrics.confusion_matrix(actual[i], predicted[i])
  cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confus_mat, display_labels=["Fail", "Pass"])
  cm_disp.plot()
  plt.show()

  print("classification_report")
  print(f"{i}.")
  print(metrics.classification_report(actual[i], predicted[i]))



print("\n--- Learned Weights (after scaling) ---")
model_list = list(models.values())
model_name_list = list(models.keys())
weight_data = {"Feature": feature_names}
for i, name in enumerate(model_name_list):
    weight_data[name] = np.round(model_list[i]["w"], 4)
print(pd.DataFrame(weight_data).to_string(index=False))

