from kedro.pipeline import Node, Pipeline

from .nodes import prepare_model_training_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=prepare_model_training_data,
                inputs="classified_text",
                outputs="model_training_data",
                name="prepare_model_training_data",
            ),
        ]
    )
