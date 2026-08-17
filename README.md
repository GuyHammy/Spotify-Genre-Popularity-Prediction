# Spotify Song Genre & Popularity Prediction

A group coursework project from my postgraduate course *Machine Learning for Data Analytics*, working with a small (~450-row) Kaggle-style dataset of Spotify songs and their audio features (BPM, energy, danceability, loudness, valence, speechiness, etc.), tackling two separate prediction problems:

- **[Genre classification](./genre_classification.ipynb)** — predict a song's top genre
- **[Popularity regression](./popularity_regression.ipynb)** — predict a song's popularity score (0-100)

This README summarizes the approach and findings for both. The full EDA, transformations, model comparisons, and reasoning live in the notebooks themselves.

## Dataset

Each song has metadata (title, artist, top genre, year) and audio features engineered by Spotify (bpm, energy, danceability, loudness `dB`, liveness, valence, duration, acousticness, speechiness, popularity). The training set is small (~450 rows) with a long tail of rare genres and mostly single-song artists, which shaped a lot of the modelling decisions below.

## Genre Classification

**EDA & feature engineering**
- Strong class imbalance: a handful of genres dominate, and 40 of 86 genres have only a single example.
- Missing genres (15 rows) were mode-imputed rather than dropped, given how small the dataset already was.
- A heatmap of artist × genre showed that almost every artist writes in a single genre — artist name turned out to be a very strong predictor, so it was one-hot encoded (345 columns) despite the resulting sparsity.
- ANOVA was used to screen numeric/engineered features (e.g. title word count) for predictive value before including them.
- Yeo-Johnson transforms were applied to right-skewed features (`title_chars`, `title_words`, `live`, `spch`).
- TF-IDF / word2vec embeddings on title and artist (with PCA) were tried but dropped — they hurt performance and added dimensionality without benefit.

**Modelling**
Because one-hot encoding the artist column produced a high-dimensional sparse feature space, linear models were favoured over tree-based ones (which struggled with the sparsity):
- Logistic Regression, Ridge Classifier, Linear SVC
- Hard Voting, Soft Voting, and Stacking ensembles built on top of the three linear models

**Results**

| Model | Accuracy |
|---|---|
| **Linear SVC** | **0.558** |
| Ridge Classifier | 0.555 |
| Hard Voting / Stacking | 0.527 |
| Soft Voting | 0.516 |

The ensembles underperformed the standalone linear models — since all three base learners are linear, voting/stacking didn't add the diversity needed to outperform the best individual model, and with only ~450 samples a stacked meta-model risked adding variance without benefit. Linear SVC was selected for the final submission, scoring **0.625** on the Kaggle-style leaderboard (vs. 0.558 in training), likely because the held-out test set skews toward the dominant, easier-to-predict genres.

## Popularity Regression

**Data cleaning & feature engineering**
- Dropped `Id` (no predictive value) and `title` (TF-IDF added no value, mostly noise).
- One-hot encoded genre/artist, keeping only the top 10 + "Other" to control dimensionality and improve generalisation to unseen categories (`handle_unknown='ignore'`).
- Log-transformed strongly right-skewed features (`spch`, `live`); left near-symmetric features (e.g. `year`) untouched to avoid introducing artifacts. `pop` itself is bimodal, so it was left unscaled for interpretability.
- Outliers in `bpm`, `dB`, `live`, `dur` were replaced with the training-set mean using IQR bounds derived only from the training data, to avoid leakage.

**Modelling**
Correlation analysis suggested non-linear relationships between features and `pop`, so tree-based models were favoured over linear regression:
- Random Forest, Extra Trees
- Stacking Regressor (RF + Extra Trees base learners, Ridge meta-model)
- Voting Regressor (soft-voting average of RF + Extra Trees)

**Results (RMSE, `pop` ranges 26-84)**

| Model | Train RMSE | Test RMSE |
|---|---|---|
| Random Forest | 10.49 | 7.78 |
| Extra Trees | 10.33 | 7.57 |
| Voting Regressor | — | 10.31 |
| **Stacking Regressor** | **9.80** | **7.05** |

The Stacking Regressor generalised best, combining Random Forest's stability with Extra Trees' variance reduction via a Ridge meta-model. Approaches that were tried and discarded: clustering, OOB estimates, gender grouping, TF-IDF on artists, PCA, XGBoost/GradientBoosting/CatBoost/neural nets (overfit or underused the features), plain linear models and SVM (couldn't capture the non-linearity).

## Key takeaways

- With very small, sparse, high-cardinality categorical data, matching model family to feature geometry (linear models for sparse one-hot spaces, tree ensembles for non-linear numeric interactions) mattered more than model sophistication.
- Ensembling only helps when base learners are genuinely diverse — stacking/voting over near-identical linear models underperformed the best single model in the classification task, while stacking two structurally different tree ensembles helped in the regression task.
- Always derive transformations (outlier bounds, imputation values, encodings) from the training set only, to avoid leaking test-set information.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn gensim
jupyter notebook genre_classification.ipynb
jupyter notebook popularity_regression.ipynb
```

Note: the raw dataset (`CS98XClassificationTrain/Test.csv`, `CS98XRegressionTrain/Test.csv`) was provided as part of the coursework and isn't redistributed here.
