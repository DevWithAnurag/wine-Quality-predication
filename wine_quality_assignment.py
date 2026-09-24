"""
Wine Quality Dataset - Data Science Lab Assignment
Dataset : winequality-red.csv  (Cortez et al., 2009)
Libraries: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn

Install (agar chahiye):  pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
Run:  python wine_quality_assignment.py   (CSV isi folder me rakho)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # Jupyter/Colab me chalana ho to ye line hata do
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
import statsmodels.api as sm

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ------------------------------------------------------------------
# LOAD DATA  (file semicolon ';' se separated hai)
# ------------------------------------------------------------------
DATA_PATH = "winequality-red.csv"       # white wine ke liye "winequality-white.csv"
df = pd.read_csv(DATA_PATH, sep=";")
features = df.columns.drop("quality").tolist()

print("Shape:", df.shape)
print(df.head(), "\n")
print(df.info(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")


# ==================================================================
# 1. PANDAS: ATTRIBUTE STATISTICS, CORRELATION, COVARIANCE,
#            INFERENTIAL STATISTICS
# ==================================================================
print("=" * 70)
print("1. DESCRIPTIVE + INFERENTIAL STATISTICS")
print("=" * 70)

# --- 1a. Attribute (descriptive) statistics ---
desc = df.describe().T
desc["median"] = df.median()
desc["variance"] = df.var()
desc["skewness"] = df.skew()
desc["kurtosis"] = df.kurt()
desc["range"] = df.max() - df.min()
desc["IQR"] = df.quantile(0.75) - df.quantile(0.25)
print("\n--- Descriptive statistics ---")
print(desc.round(4))
print("\nMode of each column:\n", df.mode().iloc[0])

# --- 1b. Correlation & Covariance ---
corr = df.corr()                         # Pearson
cov = df.cov()
print("\n--- Correlation matrix (Pearson) ---")
print(corr.round(3))
print("\n--- Covariance matrix ---")
print(cov.round(4))
print("\n--- Correlation of each feature with quality ---")
print(corr["quality"].drop("quality").sort_values(ascending=False).round(4))

plt.figure(figsize=(11, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix - Red Wine")
plt.tight_layout(); plt.savefig("fig1_correlation_heatmap.png", dpi=120); plt.close()

df[features].hist(bins=30, figsize=(14, 10))
plt.suptitle("Feature distributions"); plt.tight_layout()
plt.savefig("fig1_histograms.png", dpi=120); plt.close()

# --- 1c. Inferential statistics ---
print("\n--- Inferential statistics ---")

# (i) 95% confidence interval for mean of each attribute
print("\n(i) 95% Confidence interval for the mean:")
ci_rows = []
for c in df.columns:
    m, se = df[c].mean(), stats.sem(df[c])
    lo, hi = stats.t.interval(0.95, len(df) - 1, loc=m, scale=se)
    ci_rows.append([c, m, lo, hi])
print(pd.DataFrame(ci_rows, columns=["attribute", "mean", "CI_low", "CI_high"]).round(4).to_string(index=False))

# (ii) One-sample t-test  H0: mean(alcohol) = 10
t, p = stats.ttest_1samp(df["alcohol"], 10)
print(f"\n(ii) One-sample t-test  H0: mean alcohol = 10  ->  t = {t:.4f}, p = {p:.6f}",
      "=> Reject H0" if p < 0.05 else "=> Fail to reject H0")

# (iii) Two-sample (Welch) t-test: alcohol in good (>=6) vs bad (<6) wines
good = df[df["quality"] >= 6]["alcohol"]
bad = df[df["quality"] < 6]["alcohol"]
t, p = stats.ttest_ind(good, bad, equal_var=False)
print(f"(iii) Two-sample t-test alcohol (good vs bad wine): t = {t:.4f}, p = {p:.3e}",
      "=> Significant difference" if p < 0.05 else "=> No significant difference")

# (iv) Pearson correlation significance test with quality
print("\n(iv) Pearson r and p-value of each feature with quality:")
rows = []
for c in features:
    r, p = stats.pearsonr(df[c], df["quality"])
    rows.append([c, r, p, "Significant" if p < 0.05 else "Not significant"])
print(pd.DataFrame(rows, columns=["feature", "r", "p_value", "result(5%)"]).round(5).to_string(index=False))

# (v) One-way ANOVA: does alcohol differ across quality levels?
groups = [g["alcohol"].values for _, g in df.groupby("quality")]
F, p = stats.f_oneway(*groups)
print(f"\n(v) One-way ANOVA (alcohol across quality levels): F = {F:.3f}, p = {p:.3e}")

# (vi) Normality test (D'Agostino-Pearson)
print("\n(vi) Normality test (D'Agostino K^2):")
for c in ["alcohol", "pH", "density", "quality"]:
    k2, p = stats.normaltest(df[c])
    print(f"   {c:10s}: K2 = {k2:9.3f}, p = {p:.3e} ->", "Normal" if p > 0.05 else "Not normal")

# (vii) Chi-square test of independence: wine type vs quality? -> here quality vs alcohol-level
df["_alc_level"] = pd.qcut(df["alcohol"], 3, labels=["low", "medium", "high"])
ct = pd.crosstab(df["_alc_level"], df["quality"])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"\n(vii) Chi-square (alcohol level vs quality): chi2 = {chi2:.2f}, dof = {dof}, p = {p:.3e}")
df.drop(columns="_alc_level", inplace=True)


# ==================================================================
# 2. PCA - DIMENSIONALITY REDUCTION
# ==================================================================
print("\n" + "=" * 70)
print("2. PRINCIPAL COMPONENT ANALYSIS")
print("=" * 70)

X = df[features].values
y_reg = df["quality"].values
y_cls = (df["quality"] >= 6).astype(int).values      # binary: good / not good

X_scaled = StandardScaler().fit_transform(X)          # PCA se pehle standardize zaroori hai

# --- 2a. PCA from scratch (eigen-decomposition of covariance matrix) ---
cov_mat = np.cov(X_scaled.T)
eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
idx = np.argsort(eig_vals)[::-1]
eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]
scratch_evr = eig_vals / eig_vals.sum()
print("\nEigenvalues (scratch):", np.round(eig_vals, 4))
print("Explained variance ratio (scratch):", np.round(scratch_evr, 4))

# --- 2b. PCA using sklearn module ---
pca_full = PCA().fit(X_scaled)
evr = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr)
pca_table = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(evr))],
    "eigenvalue": pca_full.explained_variance_,
    "variance_ratio": evr,
    "cumulative": cum_evr,
})
print("\n--- sklearn PCA ---")
print(pca_table.round(4).to_string(index=False))
print("\nScratch vs sklearn variance ratio match:", np.allclose(scratch_evr, evr))

n_95 = int(np.argmax(cum_evr >= 0.95) + 1)
n_kaiser = int((pca_full.explained_variance_ > 1).sum())
print(f"\nComponents for >=95% variance : {n_95}")
print(f"Components with eigenvalue > 1 (Kaiser criterion): {n_kaiser}")

loadings = pd.DataFrame(pca_full.components_.T[:, :n_95],
                        index=features, columns=[f"PC{i+1}" for i in range(n_95)])
print("\nLoadings (feature contribution to each PC):\n", loadings.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].bar(range(1, len(evr) + 1), evr); ax[0].plot(range(1, len(evr) + 1), evr, "ro-")
ax[0].set(title="Scree plot", xlabel="Component", ylabel="Explained variance ratio")
ax[1].plot(range(1, len(evr) + 1), cum_evr, "bo-"); ax[1].axhline(0.95, color="r", ls="--")
ax[1].set(title="Cumulative explained variance", xlabel="No. of components", ylabel="Cumulative ratio")
plt.tight_layout(); plt.savefig("fig2_pca_scree.png", dpi=120); plt.close()

# --- 2c. Performance comparison: original vs PCA-reduced features ---
X_pca = PCA(n_components=n_95).fit_transform(X_scaled)

def compare(Xa, Xb, label_a, label_b):
    """Original vs PCA features par Logistic (accuracy) aur Linear (R2, RMSE) compare karo."""
    out = {}
    for name, Xm in [(label_a, Xa), (label_b, Xb)]:
        Xtr, Xte, ytr_c, yte_c, ytr_r, yte_r = train_test_split(
            Xm, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr_c)
        reg = LinearRegression().fit(Xtr, ytr_r)
        out[name] = {
            "n_features": Xm.shape[1],
            "Logistic accuracy": accuracy_score(yte_c, clf.predict(Xte)),
            "Linear R2": r2_score(yte_r, reg.predict(Xte)),
            "Linear RMSE": np.sqrt(mean_squared_error(yte_r, reg.predict(Xte))),
        }
    return pd.DataFrame(out).T

print("\n--- Performance: original features vs PCA features ---")
print(compare(X_scaled, X_pca, "Original (11)", f"PCA ({n_95})").round(4))

print("\n--- Effect of number of components on performance ---")
res = []
for k in range(1, len(features) + 1):
    Xk = PCA(n_components=k).fit_transform(X_scaled)
    Xtr, Xte, ytr, yte = train_test_split(Xk, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    res.append([k, accuracy_score(yte, LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xte))])
res = pd.DataFrame(res, columns=["n_components", "accuracy"])
print(res.round(4).to_string(index=False))

plt.figure(figsize=(6, 4.5))
plt.plot(res["n_components"], res["accuracy"], "go-")
plt.xlabel("No. of PCA components"); plt.ylabel("Logistic accuracy"); plt.title("Accuracy vs PCA components")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("fig2_pca_accuracy.png", dpi=120); plt.close()


# ==================================================================
# 3. LINEAR REGRESSION - PARAMETERS, ERROR, COMPARE WITH STANDARD MODULES
# ==================================================================
print("\n" + "=" * 70)
print("3. LINEAR REGRESSION (predict quality)")
print("=" * 70)

Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# --- 3a. From scratch: Normal Equation  theta = (X^T X)^-1 X^T y ---
def add_bias(A):
    return np.c_[np.ones(A.shape[0]), A]

Xtr_b, Xte_b = add_bias(Xtr), add_bias(Xte)
theta = np.linalg.inv(Xtr_b.T @ Xtr_b) @ Xtr_b.T @ ytr
pred_scratch = Xte_b @ theta

# --- 3b. From scratch: Gradient Descent (standardized features) ---
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = add_bias(sc.transform(Xtr)), add_bias(sc.transform(Xte))
w = np.zeros(Xtr_s.shape[1]); lr, epochs = 0.1, 2000
for _ in range(epochs):
    grad = (2 / len(ytr)) * Xtr_s.T @ (Xtr_s @ w - ytr)
    w -= lr * grad
pred_gd = Xte_s @ w

# --- 3c. Standard modules: sklearn & statsmodels ---
lr_sk = LinearRegression().fit(Xtr, ytr)
pred_sk = lr_sk.predict(Xte)

ols = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
pred_sm = ols.predict(sm.add_constant(Xte))

# --- Parameters comparison ---
params = pd.DataFrame({
    "Normal Eq (scratch)": theta,
    "sklearn": np.r_[lr_sk.intercept_, lr_sk.coef_],
    "statsmodels": ols.params,
}, index=["intercept"] + features)
print("\n--- Regression parameters (intercept + coefficients) ---")
print(params.round(5))
print("\nScratch == sklearn == statsmodels ?",
      np.allclose(params.iloc[:, 0], params.iloc[:, 1]) and np.allclose(params.iloc[:, 1], params.iloc[:, 2]))

# --- Error metrics ---
def metrics(y_true, y_pred):
    n, p = len(y_true), len(features)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2,
            "Adj R2": 1 - (1 - r2) * (n - 1) / (n - p - 1)}

err = pd.DataFrame({
    "Normal Eq (scratch)": metrics(yte, pred_scratch),
    "Gradient Descent (scratch)": metrics(yte, pred_gd),
    "sklearn": metrics(yte, pred_sk),
    "statsmodels": metrics(yte, pred_sm),
}).T
print("\n--- Error on TEST set ---")
print(err.round(5))

print("\n--- statsmodels OLS summary (train set) ---")
print(ols.summary(xname=["const"] + features))

# Residual analysis
resid = yte - pred_sk
plt.figure(figsize=(12, 4.5))
plt.subplot(1, 2, 1)
plt.scatter(yte, pred_sk, alpha=.5); plt.plot([3, 8], [3, 8], "r--")
plt.xlabel("Actual quality"); plt.ylabel("Predicted quality"); plt.title("Actual vs Predicted")
plt.subplot(1, 2, 2)
plt.scatter(pred_sk, resid, alpha=.5); plt.axhline(0, color="r", ls="--")
plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title("Residual plot")
plt.tight_layout(); plt.savefig("fig3_linear_regression.png", dpi=120); plt.close()


# ==================================================================
# 4. LOGISTIC REGRESSION - BINARY CLASSIFICATION
# ==================================================================
print("\n" + "=" * 70)
print("4. LOGISTIC REGRESSION (binary classification)")
print("=" * 70)

data = pd.read_csv(DATA_PATH, sep=";")

# --- 4a. Data preprocessing ---
print("\n[Preprocessing]")
print("Missing values         :", int(data.isnull().sum().sum()))
dups = int(data.duplicated().sum())
data = data.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed : {dups}  -> new shape {data.shape}")

# Outlier treatment: IQR winsorization (clip) on features
Q1, Q3 = data[features].quantile(0.25), data[features].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
n_out = int(((data[features] < lower) | (data[features] > upper)).sum().sum())
data[features] = data[features].clip(lower=lower, upper=upper, axis=1)
print(f"Outlier values clipped (IQR rule): {n_out}")

# Binary target: quality >= 6 -> 1 (good), else 0 (not good)
data["good"] = (data["quality"] >= 6).astype(int)
print("\nClass distribution:\n", data["good"].value_counts().rename({0: "not good (<6)", 1: "good (>=6)"}))

Xc = data[features]
yc = data["good"]

# Train/test split (stratified) then scaling (fit only on train -> no data leakage)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
scaler = StandardScaler().fit(Xc_tr)
Xc_tr_s, Xc_te_s = scaler.transform(Xc_tr), scaler.transform(Xc_te)
print(f"Train: {Xc_tr.shape},  Test: {Xc_te.shape}")

# --- 4b. Model training ---
clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xc_tr_s, yc_tr)
y_pred = clf.predict(Xc_te_s)

# --- 4c. Evaluation ---
acc = accuracy_score(yc_te, y_pred)
prec = precision_score(yc_te, y_pred)
rec = recall_score(yc_te, y_pred)
cm = confusion_matrix(yc_te, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n[Evaluation on test set]")
print(f"Accuracy  = {acc:.4f}   (TP+TN)/Total = ({tp}+{tn})/{cm.sum()}")
print(f"Precision = {prec:.4f}   TP/(TP+FP)   = {tp}/({tp}+{fp})")
print(f"Recall    = {rec:.4f}   TP/(TP+FN)   = {tp}/({tp}+{fn})")
print("\nConfusion Matrix:\n", cm)
print("\nClassification report:\n", classification_report(yc_te, y_pred, target_names=["not good", "good"]))
print(f"Train accuracy = {clf.score(Xc_tr_s, yc_tr):.4f}  |  Test accuracy = {acc:.4f}")

coef = pd.Series(clf.coef_[0], index=features).sort_values(key=abs, ascending=False)
print("\nLogistic coefficients (standardized features):\n", coef.round(4))

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["not good", "good"], yticklabels=["not good", "good"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig("fig4_confusion_matrix.png", dpi=120); plt.close()

print("\nDone. Figures saved: fig1_*, fig2_*, fig3_*, fig4_* (.png)")
