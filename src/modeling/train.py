"""Treinamento reproduzivel dos modelos, exportacao de metricas e artefato final."""

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    ARQUIVO_COMPARACAO_MODELOS,
    ARQUIVO_DADOS_BRUTOS,
    ARQUIVO_METADADOS_MODELO,
    ARQUIVO_METRICAS_MODELOS,
    ARQUIVO_MODELO,
    CLASSE_NEGATIVA,
    CLASSE_POSITIVA,
    COLUNAS_FEATURES,
    CV_FOLDS,
    MAX_DEPTH_ARVORE_CONTROLADA,
    MAX_ITER_LOGISTICA,
    NOME_MODELO_FINAL,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data.loader import (
    calcular_sha256_arquivo,
    carregar_dataset,
    dividir_treino_e_teste,
    separar_features_e_alvo,
)
from src.modeling.evaluate import (
    avaliar_no_teste,
    avaliar_validacao_cruzada,
    exportar_metricas,
)
from src.modeling.preprocessing import criar_pipeline_modelo


def definir_modelos_candidatos() -> dict:
    """Retorna os quatro modelos da metodologia: baseline, logistica e duas arvores."""
    return {
        "DummyClassifier (most_frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=MAX_ITER_LOGISTICA),
        "Decision Tree (sem limite)": DecisionTreeClassifier(random_state=RANDOM_STATE),
        f"Decision Tree (max_depth={MAX_DEPTH_ARVORE_CONTROLADA})": DecisionTreeClassifier(
            max_depth=MAX_DEPTH_ARVORE_CONTROLADA,
            random_state=RANDOM_STATE,
        ),
    }


def treinar_e_avaliar_modelos(
    caminho_dados: Path = ARQUIVO_DADOS_BRUTOS,
    caminho_modelo: Path = ARQUIVO_MODELO,
    caminho_metadados: Path = ARQUIVO_METADADOS_MODELO,
    caminho_csv_comparacao: Path = ARQUIVO_COMPARACAO_MODELOS,
    caminho_json_metricas: Path = ARQUIVO_METRICAS_MODELOS,
    salvar_artefatos: bool = True,
) -> dict:
    """Executa o fluxo completo de split, validacao cruzada, treino, teste e exportacao."""
    dados = carregar_dataset(caminho_dados)
    sha256_dataset = calcular_sha256_arquivo(caminho_dados)

    X, y = separar_features_e_alvo(dados)
    X_train, X_test, y_train, y_test = dividir_treino_e_teste(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    modelos_candidatos = definir_modelos_candidatos()
    resultados_modelos = {}
    pipeline_final = None

    for nome_modelo, estimador in modelos_candidatos.items():
        pipeline = criar_pipeline_modelo(estimador)

        metricas_cv = avaliar_validacao_cruzada(
            pipeline, X_train, y_train, cv=CV_FOLDS
        )
        pipeline.fit(X_train, y_train)
        metricas_teste = avaliar_no_teste(pipeline, X_test, y_test)

        resultados_modelos[nome_modelo] = {
            "validacao_cruzada": metricas_cv,
            "teste": metricas_teste,
        }

        if nome_modelo == NOME_MODELO_FINAL:
            pipeline_final = pipeline

    if pipeline_final is None:
        raise RuntimeError(f"Modelo final '{NOME_MODELO_FINAL}' nao foi treinado.")

    classes_modelo = [str(classe) for classe in list(pipeline_final.classes_)]
    metadados = {
        "nome_modelo": NOME_MODELO_FINAL,
        "algoritmo": type(pipeline_final.named_steps["modelo"]).__name__,
        "etapas_pipeline": list(pipeline_final.named_steps.keys()),
        "versao_projeto": "2.0.0",
        "treinado_em_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_arquivo": Path(caminho_dados).name,
        "dataset_sha256": sha256_dataset,
        "registros_totais": int(dados.shape[0]),
        "registros_treino": int(X_train.shape[0]),
        "registros_teste": int(X_test.shape[0]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "classe_positiva": CLASSE_POSITIVA,
        "classe_negativa": CLASSE_NEGATIVA,
        "classes_modelo": classes_modelo,
        "quantidade_features": len(COLUNAS_FEATURES),
        "features_esperadas": list(COLUNAS_FEATURES),
        "politica_categorias_desconhecidas": (
            "Validacao estrita com Literal no schema Pydantic da API (HTTP 422); "
            "OneHotEncoder(handle_unknown='ignore') mantido na pipeline como defesa em profundidade."
        ),
        "versoes_bibliotecas": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "metricas_validacao_cruzada": resultados_modelos[NOME_MODELO_FINAL][
            "validacao_cruzada"
        ],
        "metricas_teste_historico": resultados_modelos[NOME_MODELO_FINAL]["teste"],
        "nota_metodologica": (
            "O conjunto de teste historico (random_state=42, test_size=0.20) ja foi analisado na V1; "
            "por isso, nao foi usado para selecionar novos hiperparametros ou ajustar threshold."
        ),
    }

    if salvar_artefatos:
        caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_final, caminho_modelo)
        metadados["modelo_sha256"] = calcular_sha256_arquivo(caminho_modelo)

        caminho_metadados.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho_metadados, "w", encoding="utf-8") as arquivo:
            json.dump(metadados, arquivo, indent=2, ensure_ascii=False)

        exportar_metricas(
            resultados_modelos,
            caminho_csv=caminho_csv_comparacao,
            caminho_json=caminho_json_metricas,
        )

    return {
        "pipeline_final": pipeline_final,
        "metadados": metadados,
        "resultados_modelos": resultados_modelos,
    }


if __name__ == "__main__":
    saida = treinar_e_avaliar_modelos(salvar_artefatos=True)
    teste_lr = saida["resultados_modelos"][NOME_MODELO_FINAL]["teste"]
    cm = teste_lr["matriz_confusao"]
    print(f"Modelo final salvo em: {ARQUIVO_MODELO}")
    print(f"Metadados salvos em:   {ARQUIVO_METADADOS_MODELO}")
    print(f"Accuracy (teste):  {teste_lr['accuracy']:.6f}")
    print(f"Recall (teste):    {teste_lr['recall']:.6f}")
    print(f"Precision (teste): {teste_lr['precision']:.6f}")
    print(f"F1-score (teste):  {teste_lr['f1']:.6f}")
    print(f"Matriz de confusao: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")
