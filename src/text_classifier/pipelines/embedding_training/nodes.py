from __future__ import annotations

from math import ceil

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


def train_embedding_classifier(
    prepared_df: pd.DataFrame, parameters: dict
) -> tuple[dict, dict]:


    X = prepared_df["text"]
    y = prepared_df["label"]

    test_size = parameters["test_size"]
    random_state = parameters["random_state"]

    class_counts = y.value_counts()
    n_test_samples = ceil(len(prepared_df) * test_size)
    use_stratify = class_counts.min() >= 2 and n_test_samples >= y.nunique()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None,
    )

    embedding_model_name = parameters["embedding_model"]["name"]
    batch_size = parameters["embedding_model"]["batch_size"]
    normalize_embeddings = parameters["embedding_model"]["normalize_embeddings"]
    device = parameters["embedding_model"]["device"]

    encoder = SentenceTransformer(embedding_model_name, device=device)
    X_train_embeddings = encoder.encode(
        X_train.tolist(),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    X_test_embeddings = encoder.encode(
        X_test.tolist(),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )

    classifier = LogisticRegression(
        max_iter=parameters["logistic_regression"]["max_iter"],
        random_state=random_state,
    )
    classifier.fit(X_train_embeddings, y_train)
    predictions = classifier.predict(X_test_embeddings)

    artifact = {
        "embedding_model_name": embedding_model_name,
        "normalize_embeddings": normalize_embeddings,
        "classifier": classifier,
    }
    metrics = {
        "n_samples": int(len(prepared_df)),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "n_classes": int(y.nunique()),
        "used_stratify": use_stratify,
        "embedding_model_name": embedding_model_name,
        "embedding_dimension": int(X_train_embeddings.shape[1]),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    return artifact, metrics
