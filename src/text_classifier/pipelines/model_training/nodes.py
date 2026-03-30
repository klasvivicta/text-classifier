from __future__ import annotations

from math import ceil

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_text_classifier(
    training_df: pd.DataFrame, parameters: dict
) -> tuple[Pipeline, dict]:

    X = training_df["text"]
    y = training_df["label"]

    test_size = parameters["test_size"]
    random_state = parameters["random_state"]

    class_counts = y.value_counts()
    n_test_samples = ceil(len(training_df) * test_size)
    use_stratify = class_counts.min() >= 2 and n_test_samples >= y.nunique()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=parameters["tfidf"]["lowercase"],
                    ngram_range=tuple(parameters["tfidf"]["ngram_range"]),
                    max_features=parameters["tfidf"]["max_features"],
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=parameters["logistic_regression"]["max_iter"],
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "n_samples": int(len(training_df)),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_classes": int(y.nunique()),
        "used_stratify": use_stratify,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    print(metrics)
    
    return model, metrics
