"""Avaliacao de modelos via validacao cruzada no treino e metricas finais no teste."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate

from src.config import (
    ARQUIVO_COMPARACAO_MODELOS,
    ARQUIVO_METRICAS_MODELOS,
    CLASSE_POSITIVA,
    CV_FOLDS,
    ORDEM_CLASSES_MATRIZ,
)


def criar_dicionario_scorers() -> dict:
    """Define as metricas usadas na validacao cruzada com pos_label='Yes' e zero_division=0."""
    return {
        "accuracy": "accuracy",
        "recall": make_scorer(
            recall_score, pos_label=CLASSE_POSITIVA, zero_division=0
        ),
        "precision": make_scorer(
            precision_score, pos_label=CLASSE_POSITIVA, zero_division=0
        ),
        "f1": make_scorer(
            f1_score, pos_label=CLASSE_POSITIVA, zero_division=0
        ),
    }


def avaliar_validacao_cruzada(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = CV_FOLDS,
) -> dict:
    """Executa validacao cruzada exclusivamente sobre os dados de treino."""
    scorers = criar_dicionario_scorers()
    resultados = cross_validate(
        pipeline,
        X_train,
        y_train,
        scoring=scorers,
        cv=cv,
        return_train_score=True,
    )

    return {
        "accuracy_treino": float(resultados["train_accuracy"].mean()),
        "accuracy_validacao": float(resultados["test_accuracy"].mean()),
        "recall_validacao": float(resultados["test_recall"].mean()),
        "precision_validacao": float(resultados["test_precision"].mean()),
        "f1_validacao": float(resultados["test_f1"].mean()),
    }


def avaliar_no_teste(
    pipeline_treinada,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Calcula metricas de classificacao e matriz de confusao no conjunto de teste."""
    y_pred = pipeline_treinada.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred, labels=ORDEM_CLASSES_MATRIZ)

    tn = int(matriz[0][0])
    fp = int(matriz[0][1])
    fn = int(matriz[1][0])
    tp = int(matriz[1][1])

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(
            recall_score(
                y_test, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0
            )
        ),
        "precision": float(
            precision_score(
                y_test, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, y_pred, pos_label=CLASSE_POSITIVA, zero_division=0
            )
        ),
        "matriz_confusao": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        },
    }


def montar_tabela_comparacao(resultados_modelos: dict) -> pd.DataFrame:
    """Organiza os resultados de validacao cruzada e teste de todos os modelos em um DataFrame."""
    linhas = []
    for nome_modelo, metricas in resultados_modelos.items():
        cv = metricas["validacao_cruzada"]
        teste = metricas["teste"]
        cm = teste["matriz_confusao"]
        linhas.append(
            {
                "modelo": nome_modelo,
                "cv_accuracy_treino": round(cv["accuracy_treino"], 4),
                "cv_accuracy_validacao": round(cv["accuracy_validacao"], 4),
                "cv_recall_validacao": round(cv["recall_validacao"], 4),
                "cv_precision_validacao": round(cv["precision_validacao"], 4),
                "cv_f1_validacao": round(cv["f1_validacao"], 4),
                "teste_accuracy": round(teste["accuracy"], 4),
                "teste_recall": round(teste["recall"], 4),
                "teste_precision": round(teste["precision"], 4),
                "teste_f1": round(teste["f1"], 4),
                "TN": cm["TN"],
                "FP": cm["FP"],
                "FN": cm["FN"],
                "TP": cm["TP"],
            }
        )
    return pd.DataFrame(linhas)


def exportar_metricas(
    resultados_modelos: dict,
    caminho_csv: Path = ARQUIVO_COMPARACAO_MODELOS,
    caminho_json: Path = ARQUIVO_METRICAS_MODELOS,
) -> pd.DataFrame:
    """Salva a tabela comparativa em CSV e as metricas completas em JSON."""
    tabela = montar_tabela_comparacao(resultados_modelos)

    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(caminho_csv, index=False)

    caminho_json.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(resultados_modelos, arquivo, indent=2, ensure_ascii=False)

    return tabela
