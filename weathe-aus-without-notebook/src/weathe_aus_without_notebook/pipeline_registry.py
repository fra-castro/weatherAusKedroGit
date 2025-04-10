"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from weathe_aus_without_notebook.pipelines import data_analysis
from weathe_aus_without_notebook.pipelines import data_preparation
# Solo incluye los pipelines que estás seguro que funcionan
# from weathe_aus_without_notebook.pipelines import data_visualization

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_analysis_pipeline = data_analysis.create_pipeline()
    data_preparation_pipeline = data_preparation.create_pipeline()
    # data_visualization_pipeline = data_visualization.create_pipeline()

    return {
        "__default__": data_analysis_pipeline,
        "data_analysis": data_analysis_pipeline,
        "data_preparation": data_preparation_pipeline,
        "full": data_analysis_pipeline + data_preparation_pipeline,
    }