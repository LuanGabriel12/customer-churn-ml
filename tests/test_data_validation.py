"""Testes de carregamento, integridade e auditoria de qualidade dos dados."""

import pandas as pd
import pytest

from app.schemas import ClienteChurn
from src.config import ARQUIVO_DADOS_BRUTOS, COLUNAS_FEATURES, SHA256_DATASET_ESPERADO
from src.data.loader import (
    calcular_sha256_arquivo,
    carregar_dataset,
    dividir_treino_e_teste,
    separar_features_e_alvo,
)
from src.data.validation import auditar_qualidade_dados, exportar_relatorio_qualidade


def test_carregar_dataset_dimensoes_e_integridade():
    dados = carregar_dataset(ARQUIVO_DADOS_BRUTOS)
    conteudo_bruto = ARQUIVO_DADOS_BRUTOS.read_bytes()
    assert dados.shape == (7043, 21)
    assert b"\r\n" not in conteudo_bruto
    assert calcular_sha256_arquivo(ARQUIVO_DADOS_BRUTOS) == SHA256_DATASET_ESPERADO


def test_carregar_dataset_arquivo_inexistente(tmp_path):
    caminho_inexistente = tmp_path / "nao_existe.csv"
    with pytest.raises(FileNotFoundError):
        carregar_dataset(caminho_inexistente)


def test_carregar_dataset_colunas_ausentes_levanta_erro(tmp_path):
    csv_invalido = tmp_path / "incompleto.csv"
    pd.DataFrame({"customerID": ["0001"], "Churn": ["No"]}).to_csv(
        csv_invalido, index=False
    )
    with pytest.raises(ValueError, match="Colunas obrigatorias ausentes"):
        carregar_dataset(csv_invalido)


def test_auditoria_qualidade_dados_telco():
    dados = carregar_dataset(ARQUIVO_DADOS_BRUTOS)
    auditoria = auditar_qualidade_dados(
        dados,
        caminho_arquivo=ARQUIVO_DADOS_BRUTOS,
        campos_schema_api=set(ClienteChurn.model_fields.keys()),
    )

    assert auditoria["linhas"] == 7043
    assert auditoria["colunas"] == 21
    assert auditoria["sha256_confere"] is True
    assert auditoria["validacao_colunas"]["valido"] is True
    assert auditoria["total_valores_ausentes_explicitos"] == 0
    assert auditoria["linhas_duplicadas"] == 0
    assert auditoria["ids_duplicados"] == 0
    assert auditoria["distribuicao_alvo"]["No"]["quantidade"] == 5174
    assert auditoria["distribuicao_alvo"]["Yes"]["quantidade"] == 1869
    assert auditoria["total_charges_auditoria"]["quantidade_invalidos"] == 11
    assert auditoria["total_charges_auditoria"]["valores_brutos_invalidos"] == [" "]
    assert auditoria["total_charges_auditoria"]["tenure_nos_invalidos"] == {0: 11}
    assert auditoria["categorias_inesperadas"] == {}
    assert auditoria["consistencia_api"]["alinhado"] is True


def test_separacao_e_divisao_estratificada():
    dados = carregar_dataset(ARQUIVO_DADOS_BRUTOS)
    X, y = separar_features_e_alvo(dados)

    assert list(X.columns) == COLUNAS_FEATURES
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns

    X_train, X_test, y_train, y_test = dividir_treino_e_teste(X, y)
    assert len(X_train) == 5634
    assert len(X_test) == 1409

    taxa_treino = (y_train == "Yes").mean()
    taxa_teste = (y_test == "Yes").mean()
    assert taxa_treino == pytest.approx(taxa_teste, abs=0.002)


def test_exportar_relatorio_qualidade(tmp_path):
    saida_md = tmp_path / "data_quality_report.md"
    exportar_relatorio_qualidade(
        caminho_dados=ARQUIVO_DADOS_BRUTOS,
        caminho_saida=saida_md,
        campos_schema_api=set(ClienteChurn.model_fields.keys()),
    )
    conteudo = saida_md.read_text(encoding="utf-8")
    assert "Relatório de Qualidade dos Dados" in conteudo
    assert "TotalCharges" in conteudo
    assert "IBM Sample Data Sets" in conteudo
    assert "Apache License 2.0" in conteudo
