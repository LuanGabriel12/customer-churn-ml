"""Testes dos modulos de inteligencia comercial e fila de priorizacao de risco."""

import pytest

from src.analytics.customer_analysis import (
    analisar_segmentos_clientes,
    classificar_faixa_cobranca,
    classificar_faixa_tenure,
    gerar_relatorio_comercial_markdown,
)
from src.analytics.risk_ranking import gerar_fila_priorizacao, resumir_fila_risco
from src.config import ARQUIVO_DADOS_BRUTOS, ARQUIVO_MODELO
from src.data.loader import (
    carregar_dataset,
    dividir_treino_e_teste,
    separar_features_e_alvo,
)
from src.modeling.predict import carregar_modelo


def test_faixas_tenure_e_cobranca():
    assert classificar_faixa_tenure(0) == "01. 0 a 12 meses"
    assert classificar_faixa_tenure(12) == "01. 0 a 12 meses"
    assert classificar_faixa_tenure(24) == "02. 13 a 24 meses"
    assert classificar_faixa_tenure(48) == "03. 25 a 48 meses"
    assert classificar_faixa_tenure(72) == "04. 49 a 72 meses"

    assert classificar_faixa_cobranca(20.0) == "01. Ate US$ 35,00"
    assert classificar_faixa_cobranca(55.0) == "02. US$ 35,01 a US$ 70,00"
    assert classificar_faixa_cobranca(85.0) == "03. US$ 70,01 a US$ 90,00"
    assert classificar_faixa_cobranca(110.0) == "04. Acima de US$ 90,00"


def test_analise_comercial_totais_e_tabelas_markdown():
    dados = carregar_dataset(ARQUIVO_DADOS_BRUTOS)
    analise = analisar_segmentos_clientes(dados)

    assert analise["resumo_geral"]["total_clientes"] == 7043
    assert analise["resumo_geral"]["clientes_churn"] == 1869
    assert int(analise["tabelas"]["contrato"]["clientes"].sum()) == 7043
    assert int(analise["tabelas"]["internet"]["clientes"].sum()) == 7043
    assert int(analise["tabelas"]["tenure"]["clientes"].sum()) == 7043
    assert int(analise["tabelas"]["cobranca_mensal"]["clientes"].sum()) == 7043

    # Verifica que nenhum segmento contem '|' (que quebraria tabelas Markdown)
    for segmento in analise["tabelas"]["contrato_x_internet"]["segmento"].tolist():
        assert "|" not in segmento

    relatorio = gerar_relatorio_comercial_markdown(analise)
    assert "não comprovam relações de causa e efeito" in relatorio


def test_fila_priorizacao_ordenada_e_consistente():
    dados = carregar_dataset(ARQUIVO_DADOS_BRUTOS)
    modelo = carregar_modelo(ARQUIVO_MODELO)
    X, y = separar_features_e_alvo(dados)
    _, X_test, _, y_test = dividir_treino_e_teste(X, y)

    fila = gerar_fila_priorizacao(modelo, dados, X_test, y_real=y_test)
    assert len(fila) == 1409
    assert fila["posicao_fila"].tolist() == list(range(1, 1410))

    # Probabilidades ordenadas de forma decrescente
    probs = fila["probabilidade_churn"].tolist()
    assert probs == sorted(probs, reverse=True)

    resumo = resumir_fila_risco(fila)
    assert resumo["total_avaliados"] == 1409
    assert resumo["total_sinalizados_yes"] == 318
    assert sum(f["clientes"] for f in resumo["resumo_por_faixa"]) == 1409
    assert resumo["receita_mensal_associada_sinalizados"] == pytest.approx(25211.65)
