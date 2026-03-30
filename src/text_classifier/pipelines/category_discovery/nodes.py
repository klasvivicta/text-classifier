from __future__ import annotations

from collections import Counter

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def discover_candidate_categories(
    prepared_df: pd.DataFrame, parameters: dict
) -> tuple[pd.DataFrame, dict]:

    embedding_params = parameters["embedding_model"]
    discovery_params = parameters["discovery"]

    encoder = SentenceTransformer(
        embedding_params["name"],
        device=embedding_params["device"],
    )
    embeddings = encoder.encode(
        prepared_df["text"].tolist(),
        batch_size=embedding_params["batch_size"],
        normalize_embeddings=embedding_params["normalize_embeddings"],
        show_progress_bar=False,
    )

    dbscan = DBSCAN(
        eps=discovery_params["eps"],
        min_samples=discovery_params["min_samples"],
        metric="cosine",
    )
    cluster_labels = dbscan.fit_predict(embeddings)
    prepared_df["cluster_id"] = cluster_labels

    known_labels = sorted(prepared_df["label"].unique().tolist())
    rows: list[dict] = []
    candidate_cluster_count = 0

    for cluster_id in sorted(cluster for cluster in prepared_df["cluster_id"].unique() if cluster != -1):
        cluster_df = prepared_df[prepared_df["cluster_id"] == cluster_id].copy()
        label_counts = cluster_df["label"].value_counts()
        dominant_label = label_counts.idxmax()
        dominant_fraction = float(label_counts.max() / len(cluster_df))

        if dominant_fraction >= discovery_params["known_category_dominance_threshold"]:
            continue

        candidate_cluster_count += 1
        cluster_embeddings = embeddings[cluster_df.index.to_numpy()]
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        similarity_to_centroid = cosine_similarity(cluster_embeddings, centroid).ravel()
        representative_idx = int(similarity_to_centroid.argmax())

        text_counter = Counter()
        for text in cluster_df["text"]:
            text_counter.update(_tokenize(text))

        top_keywords = [
            token
            for token, _ in text_counter.most_common(discovery_params["top_keywords"])
        ]

        representative_text = cluster_df.iloc[representative_idx]["text"]

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_texts": int(len(cluster_df)),
                "dominant_known_label": dominant_label,
                "dominant_label_fraction": dominant_fraction,
                "candidate_new_category": True,
                "suggested_keywords": ", ".join(top_keywords),
                "representative_text": representative_text,
                "labels_in_cluster": ", ".join(
                    f"{label}:{count}" for label, count in label_counts.items()
                ),
                "sample_texts": " || ".join(cluster_df["text"].head(5).tolist()),
            }
        )

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values(
            by=["n_texts", "dominant_label_fraction"],
            ascending=[False, True],
        ).reset_index(drop=True)

    metrics = {
        "n_texts": int(len(prepared_df)),
        "known_labels": known_labels,
        "n_known_labels": int(len(known_labels)),
        "n_clusters_found": int(len([c for c in prepared_df["cluster_id"].unique() if c != -1])),
        "n_noise_points": int((prepared_df["cluster_id"] == -1).sum()),
        "n_candidate_clusters": int(candidate_cluster_count),
        "dbscan_eps": discovery_params["eps"],
        "dbscan_min_samples": discovery_params["min_samples"],
        "known_category_dominance_threshold": discovery_params[
            "known_category_dominance_threshold"
        ],
        "embedding_model_name": embedding_params["name"],
    }

    return report_df, metrics


def _tokenize(text: str) -> list[str]:
    return [
        token.strip(".,!?;:\"'()[]{}").lower()
        for token in text.split()
        if token.strip(".,!?;:\"'()[]{}")
    ]
