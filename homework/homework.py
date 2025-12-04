#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import gzip
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    median_absolute_error,
)


# Rutas base
TRAIN_PATH = "files/input/train_data.csv.zip"
TEST_PATH = "files/input/test_data.csv.zip"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"


def load_data(train_path: str, test_path: str):
    """Carga los datos de entrenamiento y prueba desde archivos .csv.zip."""
    df_train = pd.read_csv(train_path, compression="zip").copy()
    df_test = pd.read_csv(test_path, compression="zip").copy()
    return df_train, df_test


def preprocess_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Preprocesa los datos:
    - Crea la columna 'Age' = 2021 - Year.
    - Elimina 'Year' y 'Car_Name'.
    - Separa en X (features) e y (target 'Present_Price').
    """
    # Calcular edad del vehículo
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["Age"] = 2021 - df_train["Year"]
    df_test["Age"] = 2021 - df_test["Year"]

    # Eliminar columnas que ya no se usan
    df_train = df_train.drop(columns=["Year", "Car_Name"])
    df_test = df_test.drop(columns=["Year", "Car_Name"])

    # Separar variables independientes y dependiente
    X_train = df_train.drop(columns=["Present_Price"])
    y_train = df_train["Present_Price"]

    X_test = df_test.drop(columns=["Present_Price"])
    y_test = df_test["Present_Price"]

    return X_train, y_train, X_test, y_test


def build_pipeline(X_train: pd.DataFrame) -> GridSearchCV:
    """
    Construye el ColumnTransformer, el Pipeline y el GridSearchCV
    con los mismos hiperparámetros que tu código original.
    """
    # Columnas categóricas y numéricas
    categorical_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_cols", OneHotEncoder(), categorical_cols),
            ("num_cols", MinMaxScaler(), numeric_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("kbest", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    param_grid = {
        "kbest__k": range(1, 15),
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )

    return grid_search


def save_model(model, model_path: str) -> None:
    """Guarda el modelo (GridSearchCV) comprimido con gzip/pickle."""
    os.makedirs(Path(model_path).parent, exist_ok=True)
    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f)


def compute_metrics(y_true, y_pred, dataset_name: str) -> dict:
    """
    Calcula las métricas para un conjunto dado (train/test)
    con el mismo formato que tu código original.
    """
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mad": float(median_absolute_error(y_true, y_pred)),
    }


def save_metrics(metrics_list: list[dict], metrics_path: str) -> None:
    """Guarda la lista de métricas como JSONL (una línea por dict)."""
    os.makedirs(Path(metrics_path).parent, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        for m in metrics_list:
            f.write(json.dumps(m) + "\n")


def main() -> None:
    # 1. Cargar datos
    df_train, df_test = load_data(TRAIN_PATH, TEST_PATH)

    # 2. Preprocesar datos
    X_train, y_train, X_test, y_test = preprocess_data(df_train, df_test)

    # 3. Construir pipeline + GridSearchCV
    grid_search = build_pipeline(X_train)

    # 4. Entrenar modelo
    grid_search.fit(X_train, y_train)

    # 5. Guardar modelo
    save_model(grid_search, MODEL_PATH)

    # 6. Predicciones
    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)

    # 7. Calcular métricas
    metrics_train = compute_metrics(y_train, y_train_pred, "train")
    metrics_test = compute_metrics(y_test, y_test_pred, "test")

    # 8. Guardar métricas
    save_metrics([metrics_train, metrics_test], METRICS_PATH)


if __name__ == "__main__":
    main()