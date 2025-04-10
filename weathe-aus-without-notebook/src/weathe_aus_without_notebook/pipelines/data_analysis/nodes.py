import pandas as pd
import logging

logger = logging.getLogger(__name__)

def analyze_csv(data):
    """
    Analiza los datos del CSV y genera estadísticas básicas.
    
    Args:
        data: DataFrame cargado por Kedro
        
    Returns:
        Dictionary con estadísticas
    """
    # Información básica
    logger.info(f"Analizando datos de dimensiones: {data.shape}")
    
    summary = {}
    summary["shape"] = data.shape
    summary["columns"] = list(data.columns)
    summary["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
    summary["missing_values"] = data.isna().sum().to_dict()
    
    # Estadísticas numéricas
    numeric_columns = data.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        summary["numeric_stats"] = data[numeric_columns].describe().to_dict()
    
    return summary