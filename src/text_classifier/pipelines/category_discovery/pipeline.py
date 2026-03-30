from kedro.pipeline import Node, Pipeline

from .nodes import discover_candidate_categories


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=discover_candidate_categories,
                inputs=["classified_text", "params:category_discovery"],
                outputs=["candidate_category_clusters", "category_discovery_metrics"],
                name="discover_candidate_categories",
            ),
        ]
    )
