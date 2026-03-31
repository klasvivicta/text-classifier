from kedro.pipeline import Node, Pipeline

from .nodes import load_csv


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=load_csv,
                inputs="raw_classified_text",
                outputs="classified_text",
                name="load_training_data",
            ),
            Node(
                func=load_csv,
                inputs="raw_unclassified_text",
                outputs="unclassified_text",
                name="load_unclassified_data",
            ),
        ]
    )
