"""Construcao da pipeline de pre-processamento sem data leakage."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import COLUNAS_NUMERICAS, COLUNAS_PARA_ONEHOT


def corrigir_total_charges(X: pd.DataFrame) -> pd.DataFrame:
    """Converte TotalCharges para numerico dentro da pipeline; valores invalidos viram NaN."""
    X_copia = X.copy()
    X_copia["TotalCharges"] = pd.to_numeric(X_copia["TotalCharges"], errors="coerce")
    return X_copia


def criar_preprocessador() -> ColumnTransformer:
    """Cria o ColumnTransformer para variaveis numericas e categoricas."""
    tratamento_numericas = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessador = ColumnTransformer(
        transformers=[
            ("numericas", tratamento_numericas, COLUNAS_NUMERICAS),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUNAS_PARA_ONEHOT,
            ),
        ]
    )
    return preprocessador


def criar_pipeline_modelo(estimador) -> Pipeline:
    """Monta a pipeline completa unindo conversao de TotalCharges, pre-processamento e modelo."""
    conversor_total_charges = FunctionTransformer(corrigir_total_charges)
    preprocessador = criar_preprocessador()

    return Pipeline(
        [
            ("corrigir_total_charges", conversor_total_charges),
            ("preprocessador", preprocessador),
            ("modelo", estimador),
        ]
    )
