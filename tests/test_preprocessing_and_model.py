"""Testes de pre-processamento, prevencao de data leakage, modelagem e previsao."""

import pandas as pd
import pytest

from src.config import ARQUIVO_DADOS_BRUTOS, ARQUIVO_METADADOS_MODELO, ARQUIVO_MODELO
from src.data.loader import (
    calcular_sha256_arquivo,
    carregar_dataset,
    dividir_treino_e_teste,
    separar_features_e_alvo,
)
from src.modeling.predict import (
    carregar_metadados_modelo,
    carregar_modelo,
    classificar_faixa_risco,
    obter_colunas_entrada_modelo,
    obter_indices_classes,
    prever_cliente,
    prever_lote,
)
from src.modeling.preprocessing import corrigir_total_charges
from src.modeling.train import treinar_e_avaliar_modelos


def test_corrigir_total_charges_converte_e_nao_muta_original():
    df_original = pd.DataFrame({"TotalCharges": ["100.5", " ", "250"]})
    df_convertido = corrigir_total_charges(df_original)

    # Original permanece intacto
    assert df_original["TotalCharges"].tolist() == ["100.5", " ", "250"]
    assert df_convertido["TotalCharges"].iloc[0] == pytest.approx(100.5)
    assert pd.isna(df_convertido["TotalCharges"].iloc[1])
    assert df_convertido["TotalCharges"].iloc[2] == pytest.approx(250.0)


def test_treinamento_reproduz_baseline_historica_e_supera_dummy():
    saida = treinar_e_avaliar_modelos(
        caminho_dados=ARQUIVO_DADOS_BRUTOS,
        salvar_artefatos=False,
    )
    resultados = saida["resultados_modelos"]
    pipeline_final = saida["pipeline_final"]

    # 1. Reproducao exata das metricas historicas da Logistic Regression
    lr_teste = resultados["Logistic Regression"]["teste"]
    assert lr_teste["accuracy"] == pytest.approx(0.8055358410, abs=1e-6)
    assert lr_teste["recall"] == pytest.approx(0.5588235294, abs=1e-6)
    assert lr_teste["precision"] == pytest.approx(0.6572327044, abs=1e-6)
    assert lr_teste["f1"] == pytest.approx(0.6040462428, abs=1e-6)
    assert lr_teste["matriz_confusao"] == {
        "TN": 926,
        "FP": 109,
        "FN": 165,
        "TP": 209,
    }

    # 2. Modelo final supera a baseline DummyClassifier em todas as metricas
    dummy_teste = resultados["DummyClassifier (most_frequent)"]["teste"]
    assert dummy_teste["recall"] == 0.0
    assert dummy_teste["f1"] == 0.0
    assert lr_teste["accuracy"] > dummy_teste["accuracy"]
    assert lr_teste["recall"] > dummy_teste["recall"]
    assert lr_teste["f1"] > dummy_teste["f1"]

    # 3. Arvore sem limite apresenta overfitting maior que a arvore controlada
    dt_livre_cv = resultados["Decision Tree (sem limite)"]["validacao_cruzada"]
    dt_ctrl_cv = resultados["Decision Tree (max_depth=4)"]["validacao_cruzada"]
    gap_livre = dt_livre_cv["accuracy_treino"] - dt_livre_cv["accuracy_validacao"]
    gap_ctrl = dt_ctrl_cv["accuracy_treino"] - dt_ctrl_cv["accuracy_validacao"]
    assert gap_livre > 0.20
    assert gap_ctrl < gap_livre

    # 4. Prevencao de data leakage: mediana do imputer aprendida apenas no X_train
    dados = carregar_dataset(ARQUIVO_DADOS_BRUTOS)
    X, y = separar_features_e_alvo(dados)
    X_train, _, _, _ = dividir_treino_e_teste(X, y)
    mediana_tc_treino = float(
        pd.to_numeric(X_train["TotalCharges"], errors="coerce").median()
    )
    mediana_tc_base_inteira = float(
        pd.to_numeric(X["TotalCharges"], errors="coerce").median()
    )
    imputer = (
        pipeline_final.named_steps["preprocessador"]
        .named_transformers_["numericas"]
        .named_steps["imputer"]
    )
    mediana_aprendida = float(imputer.statistics_[2])
    assert mediana_aprendida == pytest.approx(mediana_tc_treino)
    assert mediana_aprendida != pytest.approx(mediana_tc_base_inteira)


def test_regressao_pipeline_salva_trata_total_charges_em_branco(dados_cliente):
    """Regressao da auditoria V1: a pipeline salva deve conter FunctionTransformer e aceitar ' '."""
    modelo = carregar_modelo(ARQUIVO_MODELO)
    assert "corrigir_total_charges" in modelo.named_steps

    cliente_bruto = dict(dados_cliente)
    cliente_bruto["tenure"] = 0
    cliente_bruto["TotalCharges"] = " "

    resultado = prever_cliente(modelo, cliente_bruto)
    assert resultado["previsao"] in ["Yes", "No"]
    assert 0.0 <= resultado["probabilidade_churn"] <= 1.0
    assert (
        resultado["probabilidade_churn"] + resultado["probabilidade_nao_churn"]
    ) == pytest.approx(1.0)


def test_identificacao_dinamica_classe_positiva_em_ordem_invertida():
    """Garante que o codigo nao presuma que 'Yes' ocupa sempre o indice 1 em predict_proba."""

    class EstimadorOrdemInvertida:
        classes_ = ["Yes", "No"]

        def predict(self, df):
            return ["Yes" for _ in range(len(df))]

        def predict_proba(self, df):
            return [[0.82, 0.18] for _ in range(len(df))]

    estimador = EstimadorOrdemInvertida()
    idx_yes, idx_no = obter_indices_classes(estimador)
    assert idx_yes == 0
    assert idx_no == 1

    registro = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
    }
    resultado = prever_lote(estimador, [registro])[0]
    assert resultado["previsao"] == "Yes"
    assert resultado["probabilidade_churn"] == pytest.approx(0.82)
    assert resultado["probabilidade_nao_churn"] == pytest.approx(0.18)


def test_classificacao_faixas_risco():
    assert classificar_faixa_risco(0.75) == "Alto"
    assert classificar_faixa_risco(0.60) == "Alto"
    assert classificar_faixa_risco(0.50) == "Moderado"
    assert classificar_faixa_risco(0.35) == "Moderado"
    assert classificar_faixa_risco(0.12) == "Baixo"


def test_correspondencia_modelo_salvo_e_metadados():
    """Verifica correspondencia estrutural e criptografica (SHA-256) entre modelo e metadados."""
    modelo = carregar_modelo(ARQUIVO_MODELO)
    metadados = carregar_metadados_modelo(ARQUIVO_METADADOS_MODELO)

    assert metadados["algoritmo"] == type(modelo.named_steps["modelo"]).__name__
    assert metadados["etapas_pipeline"] == list(modelo.named_steps.keys())
    assert metadados["classes_modelo"] == [str(c) for c in list(modelo.classes_)]
    assert metadados["features_esperadas"] == obter_colunas_entrada_modelo(modelo)
    assert metadados["modelo_sha256"] == calcular_sha256_arquivo(ARQUIVO_MODELO)
