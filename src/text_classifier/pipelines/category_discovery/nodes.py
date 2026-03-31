from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def discover_candidate_categories(
    training_df: pd.DataFrame, unclassified_df: pd.DataFrame, parameters: dict
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:

    embedding_params = parameters["embedding_model"]
    discovery_params = parameters["discovery"]
    tuning_params = parameters["tuning"]

    tuning_df = _prepare_training_rows(training_df, tuning_params)
    unclassified_df = unclassified_df.reset_index(drop=True).copy()

    encoder = SentenceTransformer(
        embedding_params["name"],
        device=embedding_params["device"],
    )

    training_embeddings = encoder.encode(
        tuning_df["text"].tolist(),
        batch_size=embedding_params["batch_size"],
        normalize_embeddings=embedding_params["normalize_embeddings"],
        show_progress_bar=False,
    )
    unclassified_embeddings = encoder.encode(
        unclassified_df["text"].tolist(),
        batch_size=embedding_params["batch_size"],
        normalize_embeddings=embedding_params["normalize_embeddings"],
        show_progress_bar=False,
    )

    tuning_results_df = _tune_dbscan_parameters(
        training_embeddings=training_embeddings,
        labels=tuning_df["label"],
        eps_values=tuning_params["eps_values"],
        min_samples_values=tuning_params["min_samples_values"],
    )

    best_result = tuning_results_df.iloc[0].to_dict()
    best_eps = float(best_result["eps"])
    best_min_samples = int(best_result["min_samples"])

    known_label_centroids = _build_label_centroids(
        labels=tuning_df["label"],
        embeddings=training_embeddings,
    )
    novelty_threshold, similarity_stats = _calibrate_novelty_threshold(
        labels=tuning_df["label"],
        embeddings=training_embeddings,
        label_centroids=known_label_centroids,
        quantile=discovery_params["novelty_similarity_quantile"],
    )

    unclassified_cluster_labels = _cluster_embeddings(
        embeddings=unclassified_embeddings,
        eps=best_eps,
        min_samples=best_min_samples,
    )
    unclassified_df["cluster_id"] = unclassified_cluster_labels

    report_df = _build_cluster_report(
        unclassified_df=unclassified_df,
        embeddings=unclassified_embeddings,
        label_centroids=known_label_centroids,
        novelty_threshold=novelty_threshold,
        top_keywords=discovery_params["top_keywords"],
    )

    metrics = {
        "embedding_model_name": embedding_params["name"],
        "training_rows_total": int(len(training_df)),
        "training_rows_used_for_tuning": int(len(tuning_df)),
        "training_rows_deduplicated": bool(
            tuning_params["deduplicate_training_texts"]
        ),
        "n_known_labels": int(training_df["label"].nunique()),
        "known_labels": sorted(training_df["label"].unique().tolist()),
        "selected_eps": best_eps,
        "selected_min_samples": best_min_samples,
        "tuning_objective_score": float(best_result["objective_score"]),
        "tuning_v_measure": float(best_result["v_measure"]),
        "tuning_coverage": float(best_result["coverage"]),
        "tuning_noise_fraction": float(best_result["noise_fraction"]),
        "tuning_clusters_found": int(best_result["n_clusters"]),
        "novelty_similarity_quantile": float(
            discovery_params["novelty_similarity_quantile"]
        ),
        "novelty_similarity_threshold": novelty_threshold,
        "known_similarity_min": similarity_stats["min"],
        "known_similarity_median": similarity_stats["median"],
        "known_similarity_max": similarity_stats["max"],
        "n_unclassified_texts": int(len(unclassified_df)),
        "n_unclassified_clusters_found": int(
            len({int(cluster_id) for cluster_id in unclassified_cluster_labels if cluster_id != -1})
        ),
        "n_unclassified_noise_points": int((unclassified_df["cluster_id"] == -1).sum()),
        "n_candidate_clusters": int(report_df["candidate_new_area"].sum())
        if not report_df.empty
        else 0,
    }

    return report_df, metrics, tuning_results_df


def _prepare_training_rows(training_df: pd.DataFrame, tuning_params: dict) -> pd.DataFrame:
    tuning_df = training_df.loc[:, ["text", "label"]].copy()
    if tuning_params["deduplicate_training_texts"]:
        tuning_df = tuning_df.drop_duplicates(subset=["text", "label"])
    return tuning_df.reset_index(drop=True)


def _tune_dbscan_parameters(
    training_embeddings: np.ndarray,
    labels: pd.Series,
    eps_values: list[float],
    min_samples_values: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    true_labels = labels.tolist()

    for eps in eps_values:
        for min_samples in min_samples_values:
            cluster_labels = _cluster_embeddings(
                embeddings=training_embeddings,
                eps=float(eps),
                min_samples=int(min_samples),
            )
            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
                true_labels,
                cluster_labels,
            )
            noise_fraction = float(np.mean(cluster_labels == -1))
            coverage = 1.0 - noise_fraction
            rows.append(
                {
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "objective_score": float(v_measure * coverage),
                    "v_measure": float(v_measure),
                    "homogeneity": float(homogeneity),
                    "completeness": float(completeness),
                    "adjusted_rand_score": float(
                        adjusted_rand_score(true_labels, cluster_labels)
                    ),
                    "coverage": coverage,
                    "noise_fraction": noise_fraction,
                    "n_clusters": int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)),
                    "n_noise_points": int(np.sum(cluster_labels == -1)),
                }
            )

    tuning_results_df = pd.DataFrame(rows).sort_values(
        by=["objective_score", "coverage", "v_measure", "homogeneity"],
        ascending=[False, False, False, False],
    )
    return tuning_results_df.reset_index(drop=True)


def _build_label_centroids(
    labels: pd.Series, embeddings: np.ndarray
) -> dict[str, np.ndarray]:
    label_centroids: dict[str, np.ndarray] = {}
    for label in sorted(labels.unique().tolist()):
        label_embeddings = embeddings[labels.to_numpy() == label]
        centroid = label_embeddings.mean(axis=0, keepdims=True)
        label_centroids[label] = _normalize_rows(centroid)[0]
    return label_centroids


def _calibrate_novelty_threshold(
    labels: pd.Series,
    embeddings: np.ndarray,
    label_centroids: dict[str, np.ndarray],
    quantile: float,
) -> tuple[float, dict[str, float]]:
    similarities = []
    for index, label in enumerate(labels.tolist()):
        similarity = cosine_similarity(
            embeddings[index : index + 1],
            label_centroids[label].reshape(1, -1),
        )[0, 0]
        similarities.append(float(similarity))

    threshold = float(np.quantile(similarities, quantile))
    similarity_stats = {
        "min": float(np.min(similarities)),
        "median": float(np.median(similarities)),
        "max": float(np.max(similarities)),
    }
    return threshold, similarity_stats


def _build_cluster_report(
    unclassified_df: pd.DataFrame,
    embeddings: np.ndarray,
    label_centroids: dict[str, np.ndarray],
    novelty_threshold: float,
    top_keywords: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for cluster_id in sorted(cluster for cluster in unclassified_df["cluster_id"].unique() if cluster != -1):
        cluster_df = unclassified_df[unclassified_df["cluster_id"] == cluster_id].copy()
        cluster_embeddings = embeddings[cluster_df.index.to_numpy()]
        centroid = _normalize_rows(cluster_embeddings.mean(axis=0, keepdims=True))
        similarity_to_centroid = cosine_similarity(cluster_embeddings, centroid).ravel()
        representative_idx = int(similarity_to_centroid.argmax())
        representative_text = cluster_df.iloc[representative_idx]["text"]

        similarity_by_label = {
            label: float(
                cosine_similarity(centroid, label_centroid.reshape(1, -1))[0, 0]
            )
            for label, label_centroid in label_centroids.items()
        }
        ranked_labels = sorted(
            similarity_by_label.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        nearest_known_label, nearest_similarity = ranked_labels[0]
        second_best_similarity = ranked_labels[1][1] if len(ranked_labels) > 1 else 0.0

        text_counter = Counter()
        for text in cluster_df["text"]:
            text_counter.update(_tokenize(text))

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_texts": int(len(cluster_df)),
                "candidate_new_area": bool(nearest_similarity < novelty_threshold),
                "nearest_known_label": nearest_known_label,
                "nearest_known_similarity": float(nearest_similarity),
                "similarity_margin_to_second_label": float(
                    nearest_similarity - second_best_similarity
                ),
                "suggested_keywords": ", ".join(
                    token for token, _ in text_counter.most_common(top_keywords)
                ),
                "representative_text": representative_text,
                "top_known_label_matches": ", ".join(
                    f"{label}:{similarity:.3f}"
                    for label, similarity in ranked_labels[:3]
                ),
                "sample_texts": " || ".join(cluster_df["text"].head(5).tolist()),
            }
        )

    report_df = pd.DataFrame(rows)
    if report_df.empty:
        return report_df

    return report_df.sort_values(
        by=["candidate_new_area", "nearest_known_similarity", "n_texts"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def _cluster_embeddings(
    embeddings: np.ndarray, eps: float, min_samples: int
) -> np.ndarray:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return dbscan.fit_predict(embeddings)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return values / safe_norms


def _tokenize(text: str) -> list[str]:
    return [
        token.strip(".,!?;:\"'()[]{}").lower()
        for token in text.split()
        if token.strip(".,!?;:\"'()[]{}")
    ]
