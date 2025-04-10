# src/weathe_aus_without_notebook/pipelines/data_preparation/nodes.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Any

logger = logging.getLogger(__name__)

def read_data():
    """
    Lee el dataset directamente para evitar problemas con el catálogo.
    
    Returns:
        pd.DataFrame: Dataset cargado
    """
    try:
        logger.info("Leyendo dataset...")
        data = pd.read_csv("data/01_raw/weatherAUS.csv", 
                          on_bad_lines='skip', 
                          encoding='utf-8')
        logger.info(f"Dataset cargado exitosamente. Dimensiones: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error al cargar dataset: {e}")
        raise

def handle_missing_values(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifica y maneja valores faltantes en el dataset.
    
    Args:
        data: DataFrame original
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame limpio y reporte de acciones
    """
    logger.info("Iniciando tratamiento de valores faltantes...")
    
    # Análisis inicial de valores faltantes
    missing_counts = data.isna().sum()
    missing_percent = (missing_counts / len(data) * 100).round(2)
    
    # Crear copia para no modificar el original
    clean_data = data.copy()
    
    # Reporte de acciones
    report = {
        "initial_shape": data.shape,
        "columns_with_nulls": missing_counts[missing_counts > 0].to_dict(),
        "null_percentages": missing_percent[missing_percent > 0].to_dict(),
        "imputation_methods": {},
        "dropped_columns": [],
        "summary": {}
    }
    
    # Estrategia para cada columna
    for column in data.columns:
        null_pct = missing_percent[column]
        
        # Si tiene más de 70% de valores faltantes, eliminar
        if null_pct > 70:
            clean_data = clean_data.drop(columns=[column])
            report["dropped_columns"].append({
                "column": column, 
                "null_percentage": float(null_pct),
                "reason": "Más del 70% de valores faltantes"
            })
            logger.info(f"Columna '{column}' eliminada ({null_pct}% valores faltantes)")
            continue
        
        # Si tiene valores faltantes, imputar según el tipo de datos
        if null_pct > 0:
            if pd.api.types.is_numeric_dtype(clean_data[column]):
                # Para variables numéricas
                median_val = clean_data[column].median()
                clean_data[column] = clean_data[column].fillna(median_val)
                report["imputation_methods"][column] = {
                    "method": "median",
                    "value": float(median_val),
                    "null_percentage": float(null_pct)
                }
                logger.info(f"Columna '{column}' imputada con mediana: {median_val}")
            else:
                # Para variables categóricas
                mode_val = clean_data[column].mode()[0]
                clean_data[column] = clean_data[column].fillna(mode_val)
                report["imputation_methods"][column] = {
                    "method": "mode",
                    "value": str(mode_val),
                    "null_percentage": float(null_pct)
                }
                logger.info(f"Columna '{column}' imputada con moda: {mode_val}")
    
    # Resumen final
    report["summary"]["initial_rows"] = data.shape[0]
    report["summary"]["initial_cols"] = data.shape[1]
    report["summary"]["final_rows"] = clean_data.shape[0]
    report["summary"]["final_cols"] = clean_data.shape[1]
    report["summary"]["cols_dropped"] = len(report["dropped_columns"])
    report["summary"]["cols_imputed"] = len(report["imputation_methods"])
    
    logger.info(f"Tratamiento de valores faltantes completado: {report['summary']['cols_imputed']} columnas imputadas, {report['summary']['cols_dropped']} columnas eliminadas")
    
    return clean_data, report

def handle_outliers(data: pd.DataFrame, missing_report: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detecta y maneja outliers en variables numéricas.
    
    Args:
        data: DataFrame después de tratar valores faltantes
        missing_report: Reporte del tratamiento de valores faltantes
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame sin outliers y reporte
    """
    logger.info("Iniciando detección y tratamiento de outliers...")
    
    # Crear copia para no modificar el original
    clean_data = data.copy()
    
    # Seleccionar solo columnas numéricas
    numeric_cols = clean_data.select_dtypes(include=['number']).columns.tolist()
    
    # Reporte de outliers
    report = {
        "methods": "IQR (Q1-1.5*IQR, Q3+1.5*IQR)",
        "columns_analyzed": numeric_cols,
        "outliers_detected": {},
        "actions_taken": {},
        "summary": {}
    }
    
    total_outliers = 0
    
    # Analizar cada columna numérica
    for column in numeric_cols:
        # Calcular límites usando método IQR
        Q1 = clean_data[column].quantile(0.25)
        Q3 = clean_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Contar outliers
        outliers_lower = (clean_data[column] < lower_bound).sum()
        outliers_upper = (clean_data[column] > upper_bound).sum()
        total_column_outliers = outliers_lower + outliers_upper
        
        if total_column_outliers > 0:
            report["outliers_detected"][column] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outliers_below": int(outliers_lower),
                "outliers_above": int(outliers_upper),
                "total_outliers": int(total_column_outliers),
                "percentage": float(round(total_column_outliers / len(clean_data) * 100, 2))
            }
            
            # Aplicar tratamiento: recortar valores a los límites (capping)
            clean_data[column] = clean_data[column].clip(lower=lower_bound, upper=upper_bound)
            
            report["actions_taken"][column] = "Capping (recorte a los límites IQR)"
            logger.info(f"Columna '{column}': {total_column_outliers} outliers tratados con capping")
            
            total_outliers += total_column_outliers
    
    # Resumen
    report["summary"]["total_numeric_columns"] = len(numeric_cols)
    report["summary"]["columns_with_outliers"] = len(report["outliers_detected"])
    report["summary"]["total_outliers_treated"] = total_outliers
    
    logger.info(f"Tratamiento de outliers completado: {total_outliers} outliers tratados en {len(report['outliers_detected'])} columnas")
    
    return clean_data, report

def standardize_data_types(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Estandariza los tipos de datos para asegurar consistencia.
    
    Args:
        data: DataFrame después de tratar outliers
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame con tipos estandarizados y reporte
    """
    logger.info("Iniciando estandarización de tipos de datos...")
    
    # Crear copia para no modificar el original
    std_data = data.copy()
    
    # Reporte de cambios
    report = {
        "original_dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
        "standardized_dtypes": {},
        "transformations": {},
        "summary": {}
    }
    
    # Conversión específica de Date a datetime64[ns]
    if 'Date' in std_data.columns:
        try:
            original_type = str(std_data['Date'].dtype)
            std_data['Date'] = pd.to_datetime(std_data['Date'])
            new_type = str(std_data['Date'].dtype)
            
            report["transformations"]['Date'] = {
                "original_type": original_type,
                "new_type": "datetime64[ns]",
                "reason": "Conversión explícita a datetime"
            }
            logger.info(f"Columna 'Date' convertida a datetime64[ns]")
        except Exception as e:
            logger.warning(f"No se pudo convertir 'Date' a datetime: {e}")
    
    # Conversión específica de Pressure9am y Pressure3pm a float64
    for col in ['Pressure9am', 'Pressure3pm']:
        if col in std_data.columns:
            try:
                original_type = str(std_data[col].dtype)
                std_data[col] = std_data[col].astype('float64')
                new_type = str(std_data[col].dtype)
                
                report["transformations"][col] = {
                    "original_type": original_type,
                    "new_type": "float64",
                    "reason": "Conversión explícita a float64"
                }
                logger.info(f"Columna '{col}' convertida a float64")
            except Exception as e:
                logger.warning(f"No se pudo convertir '{col}' a float64: {e}")
    
    # Verificar RainToday tiene valores válidos [0, 1, nan]
    if 'RainToday' in std_data.columns:
        try:
            # Verificar los valores únicos actuales
            unique_vals = std_data['RainToday'].unique()
            logger.info(f"Valores únicos en RainToday antes de conversión: {unique_vals}")
            
            # Convertir valores textuales a numéricos si es necesario
            if std_data['RainToday'].dtype == 'object':
                # Mapeo de valores textuales a numéricos
                rain_mapping = {'Yes': 1, 'No': 0}
                std_data['RainToday'] = std_data['RainToday'].map(rain_mapping)
                
                report["transformations"]['RainToday'] = {
                    "original_type": str(data['RainToday'].dtype),
                    "new_type": "float64",
                    "mapping": rain_mapping,
                    "reason": "Conversión de valores textuales a [0, 1, nan]"
                }
            
            # Asegurarse que solo tiene valores válidos (0, 1 o nan)
            mask = (std_data['RainToday'].isin([0, 1]) | std_data['RainToday'].isna())
            
            invalid_values = std_data.loc[~mask, 'RainToday'].unique()
            if len(invalid_values) > 0:
                # Convertir valores no válidos a NaN
                std_data.loc[~mask, 'RainToday'] = float('nan')
                
                if 'RainToday' not in report["transformations"]:
                    report["transformations"]['RainToday'] = {}
                    
                report["transformations"]['RainToday']["invalid_values_replaced"] = [float(v) for v in invalid_values]
                logger.info(f"Valores inválidos en RainToday convertidos a NaN: {invalid_values}")
            
            logger.info(f"Valores únicos en RainToday después de conversión: {std_data['RainToday'].unique()}")
            
        except Exception as e:
            logger.warning(f"Error al procesar RainToday: {e}")
    
    # Verificar RainTomorrow tiene valores válidos [0, 1]
    if 'RainTomorrow' in std_data.columns:
        try:
            # Verificar los valores únicos actuales
            unique_vals = std_data['RainTomorrow'].unique()
            logger.info(f"Valores únicos en RainTomorrow antes de conversión: {unique_vals}")
            
            # Convertir valores textuales a numéricos si es necesario
            if std_data['RainTomorrow'].dtype == 'object':
                # Mapeo de valores textuales a numéricos
                rain_mapping = {'Yes': 1, 'No': 0}
                std_data['RainTomorrow'] = std_data['RainTomorrow'].map(rain_mapping)
                
                report["transformations"]['RainTomorrow'] = {
                    "original_type": str(data['RainTomorrow'].dtype),
                    "new_type": "int64",
                    "mapping": rain_mapping,
                    "reason": "Conversión de valores textuales a [0, 1]"
                }
            
            # Asegurarse que solo tiene valores válidos (0, 1)
            mask = std_data['RainTomorrow'].isin([0, 1])
            
            invalid_values = std_data.loc[~mask, 'RainTomorrow'].unique()
            if len(invalid_values) > 0:
                # Para la variable objetivo, eliminamos filas con valores inválidos
                std_data = std_data[mask]
                
                if 'RainTomorrow' not in report["transformations"]:
                    report["transformations"]['RainTomorrow'] = {}
                    
                report["transformations"]['RainTomorrow']["rows_removed"] = (~mask).sum()
                report["transformations"]['RainTomorrow']["invalid_values_removed"] = [float(v) for v in invalid_values]
                logger.info(f"Se eliminaron {(~mask).sum()} filas con valores inválidos en RainTomorrow: {invalid_values}")
            
            logger.info(f"Valores únicos en RainTomorrow después de conversión: {std_data['RainTomorrow'].unique()}")
            
        except Exception as e:
            logger.warning(f"Error al procesar RainTomorrow: {e}")
    
    # Detectar automáticamente columnas de fecha (mantenemos esta parte del código original)
    date_pattern_cols = []
    for col in std_data.columns:
        if col != 'Date' and std_data[col].dtype == 'object':  # Excluimos Date porque ya lo tratamos
            # Verificar si la columna parece contener fechas
            if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower():
                date_pattern_cols.append(col)
            # O intentar convertir una muestra
            elif len(std_data[col].dropna()) > 0:
                sample_val = std_data[col].dropna().iloc[0]
                if isinstance(sample_val, str):
                    try:
                        pd.to_datetime(sample_val)
                        date_pattern_cols.append(col)
                    except:
                        pass
    
    # Estandarizar otras fechas
    for col in date_pattern_cols:
        try:
            std_data[col] = pd.to_datetime(std_data[col])
            report["transformations"][col] = {
                "original_type": str(data[col].dtype),
                "new_type": "datetime64[ns]",
                "reason": "Columna detectada como fecha"
            }
            logger.info(f"Columna '{col}' convertida a datetime")
        except Exception as e:
            logger.warning(f"No se pudo convertir '{col}' a datetime: {e}")
    
    # Reporte final
    report["standardized_dtypes"] = {col: str(dtype) for col, dtype in std_data.dtypes.items()}
    report["summary"]["total_columns"] = len(std_data.columns)
    report["summary"]["columns_transformed"] = len(report["transformations"])
    
    logger.info(f"Estandarización de tipos completada: {len(report['transformations'])} columnas transformadas")
    
    return std_data, report

def feature_engineering(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Realiza ingeniería de características para mejorar el dataset.
    
    Args:
        data: DataFrame con tipos estandarizados
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame con nuevas características y reporte
    """
    logger.info("Iniciando ingeniería de características...")
    
    # Crear copia para no modificar el original
    enhanced_data = data.copy()
    
    # Reporte de cambios
    report = {
        "features_added": {},
        "summary": {}
    }
    
    # Verificar si existe la columna Date
    if 'Date' in enhanced_data.columns and pd.api.types.is_datetime64_dtype(enhanced_data['Date']):
        # Extraer componentes de fecha
        enhanced_data['Year'] = enhanced_data['Date'].dt.year
        enhanced_data['Month'] = enhanced_data['Date'].dt.month
        enhanced_data['Day'] = enhanced_data['Date'].dt.day
        enhanced_data['DayOfWeek'] = enhanced_data['Date'].dt.dayofweek
        enhanced_data['Season'] = enhanced_data['Month'].apply(lambda m: 
            'Summer' if m in [12, 1, 2] else
            'Autumn' if m in [3, 4, 5] else
            'Winter' if m in [6, 7, 8] else 'Spring')
        
        report["features_added"]["temporal"] = {
            "source": "Date",
            "new_features": ["Year", "Month", "Day", "DayOfWeek", "Season"],
            "description": "Componentes temporales extraídos de la fecha"
        }
        logger.info("Características temporales extraídas de 'Date'")
    
    # Crear características de diferencia de temperatura
    if all(col in enhanced_data.columns for col in ['MinTemp', 'MaxTemp']):
        enhanced_data['TempRange'] = enhanced_data['MaxTemp'] - enhanced_data['MinTemp']
        report["features_added"]["temp_range"] = {
            "source": ["MinTemp", "MaxTemp"],
            "new_features": ["TempRange"],
            "description": "Rango de temperatura diario"
        }
        logger.info("Característica 'TempRange' creada")
    
    # Características de humedad
    if all(col in enhanced_data.columns for col in ['Humidity9am', 'Humidity3pm']):
        enhanced_data['HumidityChange'] = enhanced_data['Humidity3pm'] - enhanced_data['Humidity9am']
        report["features_added"]["humidity_change"] = {
            "source": ["Humidity9am", "Humidity3pm"],
            "new_features": ["HumidityChange"],
            "description": "Cambio en humedad entre mañana y tarde"
        }
        logger.info("Característica 'HumidityChange' creada")
    
    # Características de presión
    if all(col in enhanced_data.columns for col in ['Pressure9am', 'Pressure3pm']):
        enhanced_data['PressureChange'] = enhanced_data['Pressure3pm'] - enhanced_data['Pressure9am']
        report["features_added"]["pressure_change"] = {
            "source": ["Pressure9am", "Pressure3pm"],
            "new_features": ["PressureChange"],
            "description": "Cambio en presión entre mañana y tarde"
        }
        logger.info("Característica 'PressureChange' creada")
    
    # Resumen
    new_features_count = sum(len(v["new_features"]) for v in report["features_added"].values())
    report["summary"]["initial_features"] = data.shape[1]
    report["summary"]["new_features"] = new_features_count
    report["summary"]["total_features"] = enhanced_data.shape[1]
    
    logger.info(f"Ingeniería de características completada: {new_features_count} nuevas características creadas")
    
    return enhanced_data, report

def prepare_for_database(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepara los datos para ser guardados en una base de datos relacional.
    
    Args:
        data: DataFrame con características mejoradas
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame listo para base de datos y reporte
    """
    logger.info("Preparando datos para base de datos...")
    
    # Crear copia para no modificar el original
    db_data = data.copy()
    
    # Reporte de cambios
    report = {
        "renaming": {},
        "final_schema": {},
        "column_types": {},
        "summary": {}
    }
    
    # Estandarizar nombres de columnas
    original_names = db_data.columns.tolist()
    new_names = []
    
    for col in original_names:
        # Convertir a snake_case
        new_col = col.lower().replace(' ', '_')
        new_names.append(new_col)
        
        if new_col != col:
            report["renaming"][col] = new_col
    
    db_data.columns = new_names
    logger.info(f"{len(report['renaming'])} columnas renombradas a formato snake_case")
    
    # Verificar tipos y convertir para compatibilidad con BD
    for col in db_data.columns:
        # Convertir categorías a string para mejor compatibilidad
        if pd.api.types.is_categorical_dtype(db_data[col]):
            db_data[col] = db_data[col].astype(str)
            report["column_types"][col] = {"from": "category", "to": "varchar"}
        
        # Asegurar que los flotantes tengan precisión adecuada
        elif pd.api.types.is_float_dtype(db_data[col]):
            # Redondear a 3 decimales por defecto
            db_data[col] = db_data[col].round(3)
            report["column_types"][col] = {"from": str(db_data[col].dtype), "to": "numeric(10,3)"}
    
    # Crear esquema para base de datos
    for col in db_data.columns:
        col_type = db_data[col].dtype
        
        if pd.api.types.is_datetime64_dtype(col_type):
            sql_type = "timestamp"
        elif pd.api.types.is_integer_dtype(col_type):
            sql_type = "integer"
        elif pd.api.types.is_float_dtype(col_type):
            sql_type = "numeric(10,3)"
        elif pd.api.types.is_bool_dtype(col_type):
            sql_type = "boolean"
        else:
            # Para strings/objetos, determinar la longitud máxima
            max_len = db_data[col].astype(str).str.len().max()
            sql_type = f"varchar({max(50, max_len + 10)})"  # Añadir margen
        
        report["final_schema"][col] = sql_type
    
    # Generar SQL para crear tabla
    table_name = "weather_aus"
    sql_fields = []
    
    for col, col_type in report["final_schema"].items():
        sql_fields.append(f'    "{col}" {col_type}')
    
    sql_create = f'CREATE TABLE {table_name} (\n'
    sql_create += ',\n'.join(sql_fields)
    sql_create += '\n);'
    
    report["sql_create_table"] = sql_create
    
    # Resumen
    report["summary"]["total_columns"] = len(db_data.columns)
    report["summary"]["columns_renamed"] = len(report["renaming"])
    report["summary"]["table_name"] = table_name
    report["summary"]["estimated_row_size"] = sum(
        max(4, len(col) + 4) for col in db_data.columns
    )  # Estimación simple
    
    logger.info(f"Datos preparados para base de datos PostgreSQL: {len(db_data.columns)} columnas")
    
    return db_data, report

def finalize_data_preparation(
    data, missing_report, outliers_report, types_report, features_report, db_report, duplicates_report
):
    """Finaliza el proceso de preparación y genera un reporte completo."""
    logger.info("Finalizando preparación de datos...")
    
    # Crear directorio para guardar datos limpios
    import os
    
    clean_dir = "data/04_feature"
    os.makedirs(clean_dir, exist_ok=True)
    
    # Guardar datos limpios en varios formatos
    csv_path = f"{clean_dir}/weather_aus_clean.csv"
    parquet_path = f"{clean_dir}/weather_aus_clean.parquet"
    
    data.to_csv(csv_path, index=False)
    data.to_parquet(parquet_path, index=False)
    
    # Función para convertir tipos NumPy a tipos Python nativos
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        else:
            return obj
    
    # Crear reporte completo y convertir los tipos
    from datetime import datetime
    
    full_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_shape": {
            "final_rows": data.shape[0],
            "final_columns": data.shape[1]
        },
        "pipeline_steps": [
            {
                "step": "missing_values",
                "report": missing_report
            },
            {
                "step": "outlier_detection",
                "report": outliers_report
            },
            {
                "step": "data_type_standardization",
                "report": types_report
            },
            {
                "step": "feature_engineering",
                "report": features_report
            },
            {
                "step": "database_preparation",
                "report": db_report
            },
            {
                "step": "duplicates_removal", 
                "report": duplicates_report
            }
        ],
        "output_files": {
            "csv": csv_path,
            "parquet": parquet_path
        },
        "postgres_table_creation": db_report.get("sql_create_table", "")
    }
    
    # Convertir todos los tipos NumPy a tipos Python nativos
    full_report = convert_numpy_types(full_report)
    
    # Guardar reporte completo
    import json
    
    report_dir = "data/08_reporting"
    os.makedirs(report_dir, exist_ok=True)
    report_path = f"{report_dir}/data_preparation_report.json"
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=4)
    
    logger.info(f"Preparación de datos completada. Reporte guardado en: {report_path}")
    logger.info(f"Datos limpios guardados en: {csv_path} y {parquet_path}")
    
    return data

def remove_duplicates(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifica y elimina filas duplicadas del dataset.
    
    Args:
        data: DataFrame con los datos
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame sin duplicados y reporte
    """
    logger.info("Iniciando detección y eliminación de duplicados...")
    
    # Tamaño original
    original_shape = data.shape
    
    # Identificar duplicados
    duplicated_rows = data.duplicated()
    duplicated_count = duplicated_rows.sum()
    duplicated_percent = round((duplicated_count / len(data)) * 100, 2)
    
    # Crear reporte
    report = {
        "original_rows": original_shape[0],
        "duplicated_rows": int(duplicated_count),
        "duplicated_percentage": float(duplicated_percent),
        "action": "remove" if duplicated_count > 0 else "none"
    }
    
    # Eliminar duplicados solo si existen
    if duplicated_count > 0:
        data_no_dups = data.drop_duplicates().reset_index(drop=True)
        logger.info(f"Se eliminaron {duplicated_count} filas duplicadas ({duplicated_percent}%)")
        report["final_rows"] = len(data_no_dups)
        report["rows_removed"] = int(duplicated_count)
    else:
        data_no_dups = data.copy()
        logger.info("No se encontraron filas duplicadas")
        report["final_rows"] = len(data_no_dups)
        report["rows_removed"] = 0
    
    return data_no_dups, report