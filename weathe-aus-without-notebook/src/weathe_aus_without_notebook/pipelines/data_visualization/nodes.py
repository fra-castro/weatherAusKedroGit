import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def create_visualizations(data):
    """
    Crea visualizaciones para el dataset WeatherAUS.
    
    Args:
        data: DataFrame con los datos del clima de Australia
        
    Returns:
        dict: Rutas a las visualizaciones generadas
    """
    # Crear directorio para visualizaciones
    output_dir = "data/08_reporting/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar estilo de seaborn
    sns.set(style="whitegrid")
    
    # Lista para almacenar rutas de visualizaciones
    visualizations = {}
    
    # 1. Visualización de valores faltantes
    logger.info("Generando visualización de valores faltantes...")
    plt.figure(figsize=(12, 8))
    missing = data.isna().sum().sort_values(ascending=False) / len(data) * 100
    sns.barplot(x=missing.values, y=missing.index)
    plt.title('Porcentaje de valores faltantes por columna')
    plt.xlabel('% de valores faltantes')
    plt.tight_layout()
    missing_path = f"{output_dir}/missing_values.png"
    plt.savefig(missing_path)
    plt.close()
    visualizations['missing_values'] = missing_path
    
    # 2. Distribución de RainToday y RainTomorrow
    logger.info("Generando visualización de precipitaciones...")
    plt.figure(figsize=(12, 5))
    
    # Filtrar valores no nulos
    rain_data = data[['RainToday', 'RainTomorrow']].dropna()
    
    # Convertir a valores numéricos para la visualización
    rain_data_numeric = rain_data.copy()
    rain_data_numeric['RainToday'] = rain_data_numeric['RainToday'].map({'Yes': 1, 'No': 0})
    rain_data_numeric['RainTomorrow'] = rain_data_numeric['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='RainToday', data=rain_data)
    plt.title('Lluvia Hoy')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='RainTomorrow', data=rain_data)
    plt.title('Lluvia Mañana')
    
    plt.tight_layout()
    rain_dist_path = f"{output_dir}/rain_distribution.png"
    plt.savefig(rain_dist_path)
    plt.close()
    visualizations['rain_distribution'] = rain_dist_path
    
    # 3. Correlación entre variables numéricas
    logger.info("Generando matriz de correlación...")
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        correlation = data[numeric_cols].corr()
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                   linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Matriz de Correlación de Variables Numéricas')
        plt.tight_layout()
        correlation_path = f"{output_dir}/correlation_matrix.png"
        plt.savefig(correlation_path)
        plt.close()
        visualizations['correlation_matrix'] = correlation_path
    
    # 4. Distribución de temperatura por ubicación
    logger.info("Generando distribución de temperatura por ubicación...")
    plt.figure(figsize=(15, 8))
    location_temp = data.groupby('Location')['MaxTemp'].mean().sort_values(ascending=False)
    top_locations = location_temp.head(20)
    sns.barplot(x=top_locations.values, y=top_locations.index)
    plt.title('Temperatura Máxima Promedio por Ubicación (Top 20)')
    plt.xlabel('Temperatura Máxima Promedio (°C)')
    plt.tight_layout()
    temp_loc_path = f"{output_dir}/temp_by_location.png"
    plt.savefig(temp_loc_path)
    plt.close()
    visualizations['temp_by_location'] = temp_loc_path
    
    # 5. Distribución de variables climáticas principales
    logger.info("Generando distribuciones de variables climáticas...")
    climate_vars = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                   'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 
                   'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
    
    # Filtrar solo las variables que existen en el dataset
    climate_vars = [var for var in climate_vars if var in data.columns]
    
    # Crear una figura con subplots para cada variable
    n_vars = len(climate_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols  # Cálculo de filas necesarias
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, var in enumerate(climate_vars, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data[var].dropna(), kde=True)
        plt.title(f'Distribución de {var}')
        plt.tight_layout()
    
    climate_dist_path = f"{output_dir}/climate_distributions.png"
    plt.savefig(climate_dist_path)
    plt.close()
    visualizations['climate_distributions'] = climate_dist_path
    
    logger.info(f"Se han generado {len(visualizations)} visualizaciones en {output_dir}")
    return visualizations