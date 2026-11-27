"""Carregamento seguro do modelo treinado e funcoes de previsao individual e em lote."""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    ARQUIVO_METADADOS_MODELO,
    ARQUIVO_MODELO,
    CLASSE_NEGATIVA,
    CLASSE_POSITIVA,
    COLUNAS_FEATURES,
    LIMITE_RISCO_ALTO,
    LIMITE_RISCO_MODERADO,
)


def carregar_modelo(caminho_modelo: Path = ARQUIVO_MODELO):
    """Carrega a pipeline treinada a partir do arquivo joblib documentado.

    Aviso de seguranca: arquivos .joblib/.pkl utilizam serializacao pickle do Python
    e podem executar codigo arbitrario durante a desserializacao. Carregue apenas
    artefatos gerados localmente por este fluxo confiavel ou provenientes de fontes
    explicitamente confiaveis.
    """
    caminho_resolvido = Path(caminho_modelo)
    if not caminho_resolvido.exists():
        raise FileNotFoundError(
            "Artefato do modelo nao encontrado. "
            "Execute 'python -m src.modeling.train' para gerar o modelo."
        )
    return joblib.load(caminho_resolvido)


def carregar_metadados_modelo(caminho_metadados: Path = ARQUIVO_METADADOS_MODELO) -> dict:
    """Carrega o arquivo JSON de metadados verificaveis do modelo."""
    caminho_resolvido = Path(caminho_metadados)
    if not caminho_resolvido.exists():
        raise FileNotFoundError(
            "Metadados do modelo nao encontrados. "
            "Execute 'python -m src.modeling.train' para gerar os metadados."
        )
    with open(caminho_resolvido, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def obter_colunas_entrada_modelo(modelo) -> list[str]:
    """Retorna a ordem exata das colunas de entrada exigidas pelo modelo."""
    colunas_modelo = getattr(modelo, "feature_names_in_", None)
    if colunas_modelo is not None:
        return [str(coluna) for coluna in list(colunas_modelo)]
    return list(COLUNAS_FEATURES)


def obter_indices_classes(
    modelo,
    classe_positiva: str = CLASSE_POSITIVA,
    classe_negativa: str = CLASSE_NEGATIVA,
) -> tuple[int, int]:
    """Identifica dinamicamente a posicao das classes positiva e negativa em predict_proba."""
    classes = [str(classe) for classe in list(modelo.classes_)]
    if classe_positiva not in classes or classe_negativa not in classes:
        raise ValueError(
            f"Classes esperadas ('{classe_positiva}', '{classe_negativa}') "
            f"nao encontradas em modelo.classes_: {classes}"
        )
    return classes.index(classe_positiva), classes.index(classe_negativa)


def classificar_faixa_risco(probabilidade_churn: float) -> str:
    """Agrupa a probabilidade estimada em faixas operacionais de priorizacao."""
    if probabilidade_churn >= LIMITE_RISCO_ALTO:
        return "Alto"
    if probabilidade_churn >= LIMITE_RISCO_MODERADO:
        return "Moderado"
    return "Baixo"


def preparar_dataframe_entrada(registros: list[dict], modelo) -> pd.DataFrame:
    """Converte registros em DataFrame alinhado com as colunas esperadas pela pipeline."""
    if not registros:
        raise ValueError("A lista de registros para previsao nao pode estar vazia.")

    colunas_esperadas = obter_colunas_entrada_modelo(modelo)
    df = pd.DataFrame(registros)

    colunas_ausentes = [
        coluna for coluna in colunas_esperadas if coluna not in df.columns
    ]
    if colunas_ausentes:
        raise ValueError(
            f"Colunas obrigatorias ausentes na entrada de previsao: {colunas_ausentes}"
        )

    return df[colunas_esperadas]


def prever_lote(modelo, registros: list[dict]) -> list[dict]:
    """Executa previsoes de churn e probabilidades para uma lista de clientes."""
    df_entrada = preparar_dataframe_entrada(registros, modelo)
    previsoes = modelo.predict(df_entrada)
    probabilidades = modelo.predict_proba(df_entrada)
    indice_yes, indice_no = obter_indices_classes(modelo)

    resultados = []
    for indice in range(len(df_entrada)):
        resultados.append(
            {
                "previsao": str(previsoes[indice]),
                "probabilidade_churn": float(probabilidades[indice][indice_yes]),
                "probabilidade_nao_churn": float(probabilidades[indice][indice_no]),
            }
        )
    return resultados


def prever_cliente(modelo, registro_cliente: dict) -> dict:
    """Executa previsao para um unico cliente preservando o contrato da API."""
    return prever_lote(modelo, [registro_cliente])[0]
