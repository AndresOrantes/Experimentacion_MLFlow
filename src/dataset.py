# importación de librerías 
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    # traer datos de la librería 
    adult = fetch_ucirepo(id=2) 
  
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets

    if isinstance(y, pd.DataFrame):
        y = y.squeeze() # Dejamos en una dimension la y

    return X, y

def clean_target(y: pd.Series) -> pd.Series:
    # Limpiamos el Target que tiene errores de tipo
    y_clean = (
        y.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )
    y_cleaned = y_clean.map({"<=50K":0, ">50K": 1}).astype(int)

    return y_cleaned

def remove_duplicates(X: pd.DataFrame, y: pd.Series):
    # Eliminacion de filas duplicadas juntando antes X y y
    target_name = y.name if y.name else "target"

    df = pd.concat([X, y.rename(target_name)], axis=1)
    df_wo_dup= df.drop_duplicates().reset_index(drop=True)

    X_cleaned = df_wo_dup.drop(columns=[target_name])
    y_cleaned = df_wo_dup[target_name]

    return X_cleaned, y_cleaned

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float =0.2, random_state: int=42):
    # Separación en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test