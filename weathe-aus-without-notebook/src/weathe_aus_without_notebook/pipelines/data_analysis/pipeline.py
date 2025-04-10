from kedro.pipeline import Pipeline, node
from .nodes import analyze_csv

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=analyze_csv,  
                inputs="raw_data",
                outputs="data_summary",
                name="analyze_csv_node",
            ),
        ]
    )