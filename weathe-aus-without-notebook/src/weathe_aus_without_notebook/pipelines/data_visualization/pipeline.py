from kedro.pipeline import Pipeline, node
from .nodes import create_visualizations

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_visualizations,
                inputs="raw_data",
                outputs="visualizations",
                name="create_visualizations_node",
            ),
        ]
    )