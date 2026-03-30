from kedro.pipeline import Node, Pipeline

from .nodes import train_text_classifier


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_text_classifier,
                inputs=["classified_text", "params:model_training"],
                outputs=["text_classifier_model", "model_training_metrics"],
                name="train_text_classifier",
            ),
        ]
    )
