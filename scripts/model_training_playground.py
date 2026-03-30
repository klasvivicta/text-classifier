from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from definitions import ROOT_DIR 


MODEL_PATH = Path(ROOT_DIR) / "data/06_models/text_classifier_model.pkl"
METRICS_PATH = Path(ROOT_DIR) / "data/08_reporting/model_training_metrics.json"

SAMPLE_SENTENCES = [
    "Jag spelar fotboll flera gånger i veckan.",
    "Jag älskar att laga mat och baka bröd hemma.",
    "På fritiden bygger jag små programmeringsprojekt med AI.",
    "Jag vandrar gärna i fjällen och sover i tält.",
]


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run `kedro run --pipeline model_training` first."
        )

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    metrics = None
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())

    return model, metrics


def predict_sentences(model, sentences: list[str]) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(sentences)
    predictions = model.predict(sentences)
    labels = classifier.classes_

    rows = []
    for sentence, prediction, probs in zip(sentences, predictions, probabilities):
        row = {
            "text": sentence,
            "predicted_label": prediction,
            "confidence": float(probs.max()),
        }
        for label, prob in zip(labels, probs):
            row[f"prob_{label}"] = float(prob)
        rows.append(row)

    return pd.DataFrame(rows)


def inspect_tfidf(model, sentence: str, top_n: int = 15) -> pd.DataFrame:
    vectorizer = model.named_steps["tfidf"]
    transformed = vectorizer.transform([sentence])
    feature_names = vectorizer.get_feature_names_out()
    row = transformed.getrow(0)

    pairs = sorted(
        zip(row.indices, row.data),
        key=lambda item: item[1],
        reverse=True,
    )

    records = [
        {
            "feature": feature_names[index],
            "tfidf": float(value),
        }
        for index, value in pairs[:top_n]
    ]
    return pd.DataFrame(records)


def main() -> None:
    model, metrics = load_artifacts()
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    print(f"Project root: {ROOT_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Classes: {list(classifier.classes_)}")
    if metrics is not None:
        print(f"Accuracy: {metrics['accuracy']:.3f}")

    predictions_df = predict_sentences(model, SAMPLE_SENTENCES)
    print("\nPredictions")
    print(predictions_df.to_string(index=False))

    print("\nTF-IDF inspection")
    for sentence in SAMPLE_SENTENCES:
        print("\n" + "=" * 100)
        print(sentence)
        tfidf_df = inspect_tfidf(model, sentence, top_n=10)
        print(tfidf_df.to_string(index=False))

    custom_sentence = "Jag gillar att spela piano och gå på konsert."
    print("\nCustom sentence")
    print(predict_sentences(model, [custom_sentence]).to_string(index=False))
    print(inspect_tfidf(model, custom_sentence, top_n=20).to_string(index=False))


if __name__ == "__main__":
    main()
