#!/usr/bin/env python3
"""
Titanic Stacking Submission (fast, <15s)
- FE: Title, FamilySize, IsAlone, HasCabin, Deck, FarePerPerson
- Base learners: LightGBM, RandomForest
- Meta-learner: LogisticRegression on OOF preds
- Produces submission.csv with exactly len(test.csv) rows
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional but recommended: lightgbm (fast/strong). Fallback explained below if unavailable.
from lightgbm import LGBMClassifier

# --- Config ---
SEED = 42
N_SPLITS = 5

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")  # quiet LGBM name warnings
pd.options.mode.copy_on_write = True  # reduce chained assignments warnings


def read_data(root: str):
    train = pd.read_csv(os.path.join(root, "train.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))
    return train, test


# --- Feature Engineering ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Family features
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title from Name
    title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = (
        title.replace(
            ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
            "Rare",
        )
        .replace(["Mlle", "Ms"], "Miss")
        .replace("Mme", "Mrs")
        .fillna("Rare")
    )
    df["Title"] = title

    # Cabin / Deck
    has_cabin = df["Cabin"].notna().astype(int)
    deck = df["Cabin"].str[0].fillna("Unknown")
    df["HasCabin"] = has_cabin
    df["Deck"] = deck

    # Fare per person (guard against div by zero)
    df["FarePerPerson"] = (df["Fare"] / df["FamilySize"]).replace([np.inf, -np.inf], np.nan)

    # Embarked fill: mode with safe fallback
    if df["Embarked"].isna().any():
        embarked_mode = df["Embarked"].mode(dropna=True)
        df["Embarked"] = df["Embarked"].fillna(embarked_mode.iloc[0] if len(embarked_mode) else "S")

    # Drop raw text-y columns now that we’ve used them
    drop_cols = [c for c in ["Name", "Cabin", "Ticket"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def build_preprocessor(numerical_features, categorical_features) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, numerical_features), ("cat", cat_pipe, categorical_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def main():
    parser = argparse.ArgumentParser(description="Kaggle Titanic - Stacked submission")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/kaggle/input/titanic",
        help="Folder containing train.csv and test.csv (default: Kaggle path)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission.csv",
        help="Output submission CSV filename",
    )
    args = parser.parse_args()

    train_raw, test_raw = read_data(args.data_root)
    assert "Survived" in train_raw.columns, "train.csv must contain 'Survived' column"
    assert len(test_raw) > 300, f"test.csv has an unexpected row count ({len(test_raw)})."

    id_col = "PassengerId"
    label_col = "Survived"

    test_ids = test_raw[id_col].copy()
    train = train_raw.drop(columns=[id_col]).copy()
    test = test_raw.drop(columns=[id_col]).copy()

    X = train.drop(columns=[label_col])
    y = train[label_col].astype(int)

    # Feature engineering
    X = engineer_features(X)
    X_test = engineer_features(test)

    # Align columns (no row loss)
    # Ensure both train/test have identical columns and order before preprocessing
    for c in X.columns.difference(X_test.columns):
        X_test[c] = np.nan
    for c in X_test.columns.difference(X.columns):
        X[c] = np.nan
    X = X[X_test.columns]  # identical order

    # Define feature lists AFTER FE
    numerical_features = ["Age", "Fare", "FamilySize", "FarePerPerson"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "HasCabin", "Deck", "IsAlone"]

    preprocessor = build_preprocessor(numerical_features, categorical_features)

    # Base learners
    base_models = [
        (
            "lgbm",
            LGBMClassifier(
                random_state=SEED,
                n_estimators=140,
                learning_rate=0.05,
                num_leaves=20,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
                class_weight="balanced",
                verbose=-1,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                random_state=SEED,
                n_estimators=200,
                max_depth=8,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros((len(X), len(base_models)), dtype=float)
    test_meta = np.zeros((len(X_test), len(base_models)), dtype=float)

    # Train base learners with OOF predictions
    for i, (name, model) in enumerate(base_models):
        pipe = Pipeline([("pre", preprocessor), ("clf", model)])
        fold_test_preds = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            pipe.fit(X_tr, y_tr)
            oof[va_idx, i] = pipe.predict_proba(X_va)[:, 1]
            fold_test_preds.append(pipe.predict_proba(X_test)[:, 1])
        test_meta[:, i] = np.mean(fold_test_preds, axis=0)

    # Meta-learner on OOF
    meta = LogisticRegression(solver="liblinear", C=0.1, random_state=SEED)
    meta.fit(oof, y)
    oof_acc = (np.round(meta.predict_proba(oof)[:, 1]) == y.values).mean()
    print(f"OOF (meta) accuracy: {oof_acc:.4f}")

    # Final predictions
    proba = meta.predict_proba(test_meta)[:, 1]
    pred = (proba > 0.5).astype(int)

    # Build submission
    submission = pd.DataFrame({id_col: test_ids, label_col: pred})
    # Safety check: must match test.csv row count
    assert len(submission) == len(test_raw), f"Submission has {len(submission)} rows; expected {len(test_raw)}."
    # Optional: PassengerId order is unconstrained by Kaggle, but we’ll sort for readability
    submission = submission.sort_values(by=id_col).reset_index(drop=True)
    submission.to_csv(args.out, index=False)
    print(f"Saved {args.out} with {len(submission)} rows.")


if __name__ == "__main__":
    main()
