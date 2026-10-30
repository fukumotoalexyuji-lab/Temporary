# Review 1: titanic_20261027_01.ipynb

- Date: 2026-10-30
- Scope: every cell from `1. Setup` to `2.7. Appendix A.`
- I read the code and the saved outputs. I did **not** run the notebook, so the code snippets below are untested.
- The saved outputs have no errors or warnings, and the cells were run in order.

**Related files**

| File | Role |
|------|------|
| `titanic_20261027_01.ipynb` | Original notebook: baseline models (reviewed here) |
| `titanic_20261027_01_review1.md` | **This file.** Review of 01: bugs, smarter code and English |
| `titanic_20261027_01_claude1.ipynb` | Cleaned-up version of 01, based on this review (same CV scores, XGBoost bug fixed) |

Flow: 01 → review1 → 01_claude1

---

## Summary

| # | Type | Where | What | Priority |
|---|------|-------|------|----------|
| B1 | Bug | 2.5 XGBoost | Predicts with `model_lgb` instead of `model_xgb` | **High** |
| B2 | Bug | 2.4 LightGBM, 2.5 XGBoost | Submission file name is `submit_baseline_hgb` | Medium |
| B3 | Bug | 2.4 LightGBM, 2.5 XGBoost | The "skipped" message says "HistGradientBoosting" | Low |
| B4 | Bug | Setup / 2.2 RF | `N_TREES = 500` is defined but RF uses `n_estimators=100` | Medium |
| B5 | Bug | 2.1–2.6 | "Survival Rate (actual - predicted)" is not the actual rate | Low |
| B6 | Bug | Setup | `N_BOOST` comment says "LightGBM / HGB" | Low |
| S1 | Smarter code | 2.1–2.6 | Six almost identical cells → one helper function | **High** |
| S2–S8 | Smarter code | various | Small improvements | Low |
| E | English | various | Headings, comments, print text | — |

B1, B2 and B3 come from the same cause: a cell was copied and some names were not changed. S1 removes this cause.

---

## 1. Bugs

### B1 (High): XGBoost uses the LightGBM model for prediction

Cell 2.5:

```python
model_xgb.fit(X, y)                        # refit on all train data
predictions = model_lgb.predict(X_test)    # for submission only   <- model_lgb!
```

- The CV score is correct, because `cross_val_score` uses `model_xgb`.
- The **predictions are wrong**: they come from LightGBM. If `is_MK_SUBMISSION = True`, the "XGBoost" submission is really a LightGBM submission.
- The proof is the code itself. The output (0.3565) is the same as LightGBM, but it is also the same as RF, so the output alone does not prove it.

Fix:

```python
predictions = model_xgb.predict(X_test)    # for submission only
```

### B2 (Medium): Wrong submission file name in LightGBM and XGBoost

Both cells use the HGB name:

```python
submit_path = os.path.join(OUT_DIR, mk_fname("submit_baseline_hgb", directory=OUT_DIR))
```

`mk_fname` adds a serial number, so no file is overwritten. But you cannot tell which model made which file. Use `"submit_baseline_lgb"` and `"submit_baseline_xgb"`.

### B3 (Low): Wrong "skipped" message in LightGBM and XGBoost

```python
print(style("HistGradientBoosting for baseline is skipped", ...))
```

Use "LightGBM ..." and "XGBoost ...".

### B4 (Medium): `N_TREES` is not used

Setup says:

```python
N_TREES      = 500      # RandomForest
```

But cell 2.2 uses `n_estimators=100`. Choose one:

- Use the constant: `n_estimators=N_TREES`. **Note:** this changes the RF CV score (0.7912), so record the new score.
- Or delete `N_TREES` from Setup.

### B5 (Low): "actual" is not the actual survival rate

```python
print(f"Survival Rate (actual - predicted):  {(survival_rate - predictions.mean()):.4f}")
```

`survival_rate` is the rate in **train**. The actual rate in **test** is unknown, because test has no `Survived` column. Suggested label:

```python
print(f"Survival Rate (train - predicted): {survival_rate - predictions.mean():.4f}")
```

(All six models give a positive value, about 0.03. This means they predict fewer survivors than the train rate. That is common with a 0.5 threshold, so it is not a problem.)

### B6 (Low): Old comment on `N_BOOST`

```python
N_BOOST      = 100      # LightGBM / HGB
```

HGB uses `HGB_MAX_ITER`. `N_BOOST` is used by LightGBM and XGBoost. Suggested: `# LightGBM / XGBoost`.

---

## 2. Smarter coding

### S1 (High): Replace the six model cells with one helper function

Cells 2.1–2.6 repeat the same steps: features → `get_dummies` → CV → fit → predict → print → submission. When code is copied six times, small mistakes appear (B1, B2, B3). Also, when you add a new feature later, you must change six places.

Put these functions in `1. Setup` (after `survival_rate` is defined, or move them to 2.):

```python
BASE_FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]

def make_xy(features, dummy_cols=None):
    """One-hot encode train/test the same way and align test columns to train."""
    X      = pd.get_dummies(train[features], columns=dummy_cols)
    X_test = pd.get_dummies(test[features],  columns=dummy_cols)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    return X, X_test


def run_baseline(key, label, model, X, X_test, fit_params=None):
    """CV score -> refit on all training data -> predict test -> (optional) save submission."""
    y = train[TARGET]
    fit_params = fit_params or {}

    scores = cross_val_score(model, X, y, cv=CV, scoring=SCORING, params=fit_params)
    print(f"{label} CV accuracy : {scores.mean():.4f} +/- {scores.std():.4f}")

    model.fit(X, y, **fit_params)                      # refit on all training data
    predictions = model.predict(X_test).astype(int)    # for submission only
    print(f"Survival Rate (predicted)        : {predictions.mean():.4f}")
    print(f"Survival Rate (train - predicted): {survival_rate - predictions.mean():.4f}")

    if is_MK_SUBMISSION:
        submission  = pd.DataFrame({ID: test[ID], TARGET: predictions})
        submit_path = os.path.join(OUT_DIR, mk_fname(f"submit_baseline_{key}", directory=OUT_DIR))
        submission.to_csv(submit_path, index=False)
        print("Saved:", submit_path)

    return model, predictions
```

Then each model cell is only a few lines. You can keep one cell per model under the same headings:

```python
X, X_test       = make_xy(BASE_FEATURES)
X_lr, X_test_lr = make_xy(BASE_FEATURES, dummy_cols=["Pclass", "Sex"])

models, preds = {}, {}      # keep every model and its predictions
```

```python
# 2.1. Logistic Regression
if is_LR_base:
    models["lr"], preds["lr"] = run_baseline(
        "lr", "Logistic Regression",
        LogisticRegression(max_iter=LR_MAX_ITER),
        X_lr, X_test_lr)
```

```python
# 2.5. XGBoost
if is_XGB_base:
    models["xgb"], preds["xgb"] = run_baseline(
        "xgb", "XGBoost",
        XGBClassifier(learning_rate=LEARNING_RATE, n_estimators=N_BOOST,
                      random_state=RANSEED, verbosity=0),
        X, X_test)
```

```python
# 2.6. CatBoost (no get_dummies; cat_features is passed to fit)
if is_CAT_base:
    models["cat"], preds["cat"] = run_baseline(
        "cat", "CatBoost",
        CatBoostClassifier(iterations=N_CAT, learning_rate=LEARNING_RATE, random_seed=RANSEED,
                           verbose=0, allow_writing_files=False),
        train[BASE_FEATURES], test[BASE_FEATURES],
        fit_params={"cat_features": ["Sex"]})
```

RF, HGB and LightGBM follow the same pattern. Benefits:

1. **B1 cannot happen.** The same `model` object is used for CV, fit and predict.
2. **B2 cannot happen.** The file name comes from `key`.
3. **One place for features.** Change `BASE_FEATURES` once when you add `Age_missing`, `Has_Cabin`, `Title` and so on.
4. **`reindex` is applied for every model**, not only for LR.
5. **The `models` and `preds` dicts keep all results.** Today each cell overwrites `predictions`, `X` and `X_test`. With the dicts, you can compare or ensemble predictions later, and the Appendix does not depend on "whichever cell ran last" (see S2).
6. `.astype(int)` makes sure the submission has `0`/`1`, not `0.0`/`1.0`, for every model.

### S2: Make the Appendix safe when a model is skipped

The Appendix uses `model_rf`, `model_lgb`, `model_xgb` and `model_cat`. If one of the `is_*_base` flags is `False`, you get a `NameError`, or worse, an old model from an earlier run. With the `models` dict from S1:

```python
needed = ["rf", "lgb", "xgb", "cat"]
missing_models = [k for k in needed if k not in models]

if missing_models:
    print("Appendix skipped. Not trained:", missing_models)
else:
    importance = pd.DataFrame({
        "RF":       models["rf"].feature_importances_,
        "LightGBM": models["lgb"].booster_.feature_importance(importance_type="gain"),
        "XGBoost":  models["xgb"].feature_importances_,
    }, index=models["rf"].feature_names_in_)
    # ... (rest is the same)
```

### S3: Keep `survival_rate` unrounded

```python
survival_rate = train[TARGET].mean().round(4)
```

Rounding here makes every later calculation slightly less exact. Round only when you print:

```python
survival_rate = train[TARGET].mean()
print(f"Survival Rate (overall): {survival_rate:.4f}")
```

The comment `# another way of coding: round(train[TARGET].mean(), 3))` is now old (it says 3 digits and has an extra `)`). You can delete it, because cell `73abeecf` already explains this.

### S4: `missing_ratio` – multiply first, then round

```python
"missing_ratio": df.isna().mean().round(3) * 100,
```

Rounding before `* 100` can give values like `77.10000000000001`. This is cleaner:

```python
"missing_ratio": (df.isna().mean() * 100).round(1),
```

(The `test / missing` column shows `327.0` instead of `327` because test has no `Survived` row, so pandas fills `NaN` and changes the column to float. This is OK.)

### S5: Simpler check for the sample submission

```python
assert (sample[ID].values == test[ID].values).all()
```

This works, but if the lengths are different, the error is not clear. `equals` checks the length, the order and the values in one call:

```python
assert sample[ID].equals(test[ID]), "sample submission and test.csv have different PassengerIds"
```

### S6: Use `ID` everywhere

Submission cells use `test["PassengerId"]`. You already have `ID = "PassengerId"`, so use `test[ID]` (S1 does this).

### S7: Remove unused imports

`time`, `train_test_split`, `cross_val_predict` and `accuracy_score` are not used now. `warnings` is used only in a commented-out line. Remove them, or keep them only if you plan to use them soon.

### S8: Small clean-ups

- 2.6 CatBoost: the `# <- changed: ...` markers were for the review of that change. You can remove them now.
- 2.5 XGBoost: `# make submission file` starts at column 0, but it is inside the `if` block. Indent it like the other cells (it works, but it is hard to read).
- Setup: the comments are not aligned (`HGB_MAX_ITER = 1000    #`, `N_CAT       = 200      #`). Small thing, but aligned comments are easier to scan.
- `## 0. Contents` is empty. Add the list of sections, or remove the heading.

---

## 3. English

| Where | Current | Suggested | Why |
|-------|---------|-----------|-----|
| 1.1 code | `# Never change original data 0 series` | `# Never modify the original data (the *_0 variables)` | "0 series" is unclear. Say which variables. |
| 1.1 code | `# check whether submission sample is the same as test.csv` | `# Check that the sample submission has the same PassengerIds as test.csv` | Say exactly what is checked. |
| Setup | `#  TRAIN_PATH = f"{DATA_DIR}/train.csv" is also available` | `# TRAIN_PATH = f"{DATA_DIR}/train.csv" would also work` | "available" is for things you can get or use. For another way to write code, say "would also work". |
| Setup | `# Just for my study` | `# For reference: library versions` | Natural for a note to yourself. |
| Setup | `# HistGradientBoost`, `# Cat boost` | `# HistGradientBoosting`, `# CatBoost` | Use the real library names. |
| 1.2 heading | `#### 1.2. Survey data overview` | `#### 1.2. Data overview` | "survey" usually means a questionnaire. |
| 1.2 code | `print("train.describes()")` | `print("train.describe()")` | The method name has no "s". |
| 1.3 heading | `#### 1.3. Survey missing data` | `#### 1.3. Check missing data` | Same as above. |
| 1.3 note | `> Imputing Cabin seems to be very hard but it could be useful` | `> Imputing Cabin seems very hard, but it could still be useful.` | Put a comma before "but" when it joins two full sentences. "to be" is not needed. |
| 1.3 note | `> See EDA` | `> See the EDA notebook.` | Add "the" and say which EDA. |
| 1.4 heading | `#### 1.4. Survey survival rate ( overall)` | `#### 1.4. Overall survival rate` | Shorter. Also no space after "(". |
| 1.4 code | `# Tips something.round(3) and round(something, 3)` | `# Tip: x.round(3) vs. round(x, 3)` | One tip → "Tip:". "vs." shows a comparison. |
| 1.5 heading | `#### 1.5 Categorize` | `#### 1.5. Group columns by type` | "Categorize" alone does not say what. Also add "." after 1.5, like the other headings. |
| 1.5 code | `# insert numerical column to list and drop ID, TARGET` | `# Collect numerical columns (excluding ID and TARGET)` | "insert" means "put into the middle of something". You are collecting names into a list. |
| 1.5 code | `# insert category column to list` | `# Collect categorical columns` | Same. The usual term is "categorical". |
| 1.5 code | `"Category columns: "` | `"Categorical columns: "` | Same. |
| 1.5 code | `"Drop 'Name' since no distribution"` | `"Drop 'Name' (unique for every passenger)"` | Says the real reason: every value is different. |
| 2 heading | `## 2. Make baseline` | `## 2. Baseline models` | A heading is usually a noun phrase. You make several baselines, so use the plural. |
| 2.1–2.6 code | `# refit on all train data` | `# refit on all training data` | "training data" is the usual term. |
| 2.1 note | `> result` | `> Results` | Two results are listed, so use the plural. Start with a capital letter. |
| 2.1 note | `` `petit test for reindex` `` | `` Quick test: when does `reindex` matter? `` | "petit" is French. "quick test" is natural in English. |
| 2.1 note | `When does reindex work valid` | (merge into the line above) | "work valid" is not grammatical. "When is reindex useful?" or "When does reindex matter?" is natural. |
| 2.1 code | `"Logistic Regression for Baseline is skipped"` | `"Logistic Regression baseline is skipped"` | Shorter. Also, other cells use lowercase "baseline". |
| 2.1–2.6 code | `Survival Rate (actual - predicted)` | `Survival Rate (train - predicted)` | See B5: it is the train rate, not the actual test rate. |

---

## 4. Suggested order of work

1. Fix **B1** now, before you make any XGBoost submission.
2. Fix **B2** and **B3** (rename only).
3. Decide about **B4** (use `N_TREES` or delete it).
4. Do **S1** + **S2** when you start feature engineering. That is when the duplicated code will start to hurt.
5. Fix the English when you have time.
