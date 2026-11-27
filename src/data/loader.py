"""Carregamento reproduzivel do dataset e separacao entre treino e teste."""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ARQUIVO_DADOS_BRUTOS,
    COLUNA_ALVO,
    COLUNA_ID,
    COLUNAS_OBRIGATORIAS_DATASET,
    RANDOM_STATE,
    TEST_SIZE,
)


def calcular_sha256_arquivo(caminho: Path = ARQUIVO_DADOS_BRUTOS) -> str:
    """Calcula o hash SHA-256 de um arquivo para verificar integridade."""
    caminho_resolvido = Path(caminho)
    if not caminho_resolvido.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_resolvido}")

    hasher = hashlib.sha256()
    with open(caminho_resolvido, "rb") as arquivo:
        bloco = arquivo.read(65536)
        while bloco:
            hasher.update(bloco)
            bloco = arquivo.read(65536)
    return hasher.hexdigest()


def carregar_dataset(caminho: Path = ARQUIVO_DADOS_BRUTOS) -> pd.DataFrame:
    """Carrega o CSV bruto de churn sem alterar o arquivo original."""
    caminho_resolvido = Path(caminho)
    if not caminho_resolvido.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em '{caminho_resolvido}'. "
            "Verifique se o arquivo WA_Fn-UseC_-Telco-Customer-Churn.csv esta em data/."
        )

    dados = pd.read_csv(caminho_resolvido)
    colunas_faltantes = [
        coluna
        for coluna in COLUNAS_OBRIGATORIAS_DATASET
        if coluna not in dados.columns
    ]
    if colunas_faltantes:
        raise ValueError(
            f"Dataset invalido. Colunas obrigatorias ausentes: {colunas_faltantes}"
        )

    return dados


def separar_features_e_alvo(dados: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features X (sem customerID e Churn) e a variavel alvo y."""
    colunas_para_remover = [COLUNA_ALVO]
    if COLUNA_ID in dados.columns:
        colunas_para_remover.append(COLUNA_ID)

    X = dados.drop(columns=colunas_para_remover).copy()
    y = dados[COLUNA_ALVO].copy()
    return X, y


def dividir_treino_e_teste(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide X e y em treino e teste preservando a proporcao da classe alvo (stratify)."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
