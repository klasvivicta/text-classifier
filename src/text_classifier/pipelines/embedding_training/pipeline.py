from kedro.pipeline import Node, Pipeline

from .nodes import train_embedding_classifier


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_embedding_classifier,
                inputs=["classified_text", "params:embedding_training"],
                outputs=["embedding_text_classifier_artifact", "embedding_training_metrics"],
                name="train_embedding_classifier",
            ),
        ]
    )
