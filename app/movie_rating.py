"""Train and serve a reproducible movie-rating regression model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Rating_10"
NUMERIC_FEATURES = [
    "Duration_Min",
    "Budget_Crore_INR",
    "Popularity_Index",
    "Release_Year",
    "Review_Count",
]
CATEGORICAL_FEATURES = ["Genre", "Release_Platform"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the source CSV, converting malformed numeric cells to missing."""
    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[TARGET]).copy()
    if frame.empty:
        raise ValueError("Dataset contains no rows with a valid rating target")
    if not frame[TARGET].between(0, 10).all():
        raise ValueError("Rating_10 values must be between 0 and 10")
    return frame


def build_model() -> Pipeline:
    """Build a preprocessing + regression pipeline; all preprocessing is fit on train data."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_evaluate(data: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    """Fit on a reproducible holdout split and return quality metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).clip(0, 10)
    return model, {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
    }


def predict_rating(model: Pipeline, movie: dict[str, Any]) -> float:
    """Predict a rating for one movie record, clamped to the valid 0–10 range."""
    row = pd.DataFrame([{feature: movie.get(feature) for feature in FEATURES}])
    for column in NUMERIC_FEATURES:
        row[column] = pd.to_numeric(row[column], errors="coerce")
    prediction = float(model.predict(row)[0])
    return round(max(0.0, min(10.0, prediction)), 2)
