# src/weathe_aus_without_notebook/pipelines/data_preparation/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import (
    read_data,
    handle_missing_values,
    handle_outliers,
    standardize_data_types,
    feature_engineering,
    prepare_for_database,
    remove_duplicates,
    finalize_data_preparation
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=read_data,
                inputs=None,
                outputs="raw_data_memory",
                name="read_weather_data_node",
            ),
            node(
                func=handle_missing_values,
                inputs="raw_data_memory",
                outputs=["data_no_missing", "missing_report"],
                name="handle_missing_values_node",
            ),
            node(
                func=handle_outliers,
                inputs=["data_no_missing", "missing_report"],
                outputs=["data_no_outliers", "outliers_report"],
                name="handle_outliers_node",
            ),
            node(
                func=standardize_data_types,
                inputs="data_no_outliers",
                outputs=["data_standardized", "types_report"],
                name="standardize_data_types_node",
            ),
            node(
                func=feature_engineering,
                inputs="data_standardized",
                outputs=["data_with_features", "features_report"],
                name="feature_engineering_node",
            ),
            node(
                func=prepare_for_database,
                inputs="data_with_features",
                outputs=["data_for_db", "db_report"],
                name="prepare_for_database_node",
            ),
            node(
                func=remove_duplicates,
                inputs="data_for_db",
                outputs=["data_no_duplicates", "duplicates_report"],
                name="remove_duplicates_node",
            ),
            node(
                func=finalize_data_preparation,
                inputs=[
                    "data_no_duplicates",  # Cambiado de data_for_db a data_no_duplicates 
                    "missing_report", 
                    "outliers_report", 
                    "types_report", 
                    "features_report", 
                    "db_report",
                    "duplicates_report"
                ],
                outputs="final_clean_data",
                name="finalize_data_preparation_node",
            ),
        ]
    )