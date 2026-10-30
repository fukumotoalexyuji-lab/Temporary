# Kaggle Titanic: baseline models

Baseline models for the Kaggle competition
[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

This repository also shows **how I work with AI**:
I write the code myself, ask an AI assistant ([Claude](https://www.anthropic.com/claude) by Anthropic) to review it,
and then compare my version with the improved one.

## Workflow

```
01 (my notebook)  →  review1 (AI review)  →  01_claude1 (cleaned-up version)
```

| File | What it is |
|------|------------|
| [`titanic_20261027_01.ipynb`](titanic_20261027_01.ipynb) | **My notebook.** Data check and six baseline models |
| [`titanic_20261027_01_review1.md`](titanic_20261027_01_review1.md) | AI review of 01: bugs, smarter code and English |
| [`titanic_20261027_01_claude1.ipynb`](titanic_20261027_01_claude1.ipynb) | Cleaned-up version of 01, based on the review |
| [`mylib/myutils.py`](mylib/myutils.py) | Small helpers used by both notebooks (file names, separators, colored text) |

### What the review found

- **A bug I had missed:** the XGBoost cell predicted with the LightGBM model.
  The CV score was correct, but the predictions were not.
  This and two smaller mistakes came from copying the same cell six times.
- I fixed these three bugs in 01 (marked with a comment), so 01 now gives correct results.
- 01_claude1 removes the cause: one helper function, `run_baseline()`, is used for all six models.
  It gives the same CV scores as 01.

## Results

5-fold stratified cross-validation, accuracy.
Features: `Pclass`, `Sex`, `SibSp`, `Parch` (no feature engineering yet).

| Model | CV accuracy |
|-------|-------------|
| Logistic Regression | 0.7969 ± 0.0213 |
| Random Forest | 0.7912 ± 0.0171 |
| HistGradientBoosting | 0.7912 ± 0.0121 |
| XGBoost | 0.7834 ± 0.0210 |
| LightGBM | 0.7812 ± 0.0140 |
| CatBoost | 0.7733 ± 0.0268 |

The gap between the best and the worst model (0.024) is about the same size as the CV std,
so the six models perform about the same.
`Sex` is the most important feature, followed by `Pclass` (see Appendix A in each notebook).

## How to run

1. Download `train.csv`, `test.csv` and `gender_submission.csv` from the
   [competition page](https://www.kaggle.com/competitions/titanic/data)
   and put them in `data/titanic/`.
   The data is not included in this repository.
2. Install the libraries (tested with these versions):

   ```bash
   pip install pandas==2.2.3 numpy==2.1.3 lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10 "scikit-learn>=1.4" matplotlib jupyter
   ```

3. Open a notebook and run all cells. Tested with Python 3.13.

```
.
├── data/titanic/           # Kaggle data (not included)
├── mylib/myutils.py
├── titanic_20261027_01.ipynb
├── titanic_20261027_01_review1.md
└── titanic_20261027_01_claude1.ipynb
```

## Next steps

- Feature engineering: `Age` (with missing values), `Fare`, `Embarked`, title from `Name`, cabin deck
- Hyperparameter tuning and ensembling
