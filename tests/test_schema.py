"""Testes de validacao dos schemas Pydantic (preservados e corrigidos da API local)."""

from pydantic import ValidationError
import pytest

from app.schemas import ClienteChurn, LoteClientesChurn
from src.config import COLUNAS_FEATURES, LIMITE_MAXIMO_LOTE


def test_senior_citizen_invalido(dados_cliente):
    """Regressao corrigida da V1: payload completo com apenas SeniorCitizen invalido."""
    dados_cliente["SeniorCitizen"] = 2

    with pytest.raises(ValidationError) as exc_info:
        ClienteChurn.model_validate(dados_cliente)

    erros = exc_info.value.errors()
    assert len(erros) == 1
    assert erros[0]["loc"] == ("SeniorCitizen",)
    assert erros[0]["type"] == "less_than_equal"


def test_quantidade_features():
    assert len(ClienteChurn.model_fields) == 19


def test_nomes_features_alinhados_com_treinamento():
    assert set(ClienteChurn.model_fields.keys()) == set(COLUNAS_FEATURES)


@pytest.mark.parametrize("teste_tenure", [0, 1, 12, 72])
def test_schema_tenure_valores_validos(dados_cliente, teste_tenure):
    """Regressao corrigida da V1: usa atribuicao '=' (e nao '==') para testar cada valor."""
    dados_cliente["tenure"] = teste_tenure
    cliente = ClienteChurn.model_validate(dados_cliente)
    assert cliente.tenure == teste_tenure


def test_schema_tenure_string_numerica_convertida(dados_cliente):
    dados_cliente["tenure"] = "12"
    cliente = ClienteChurn.model_validate(dados_cliente)
    assert cliente.tenure == 12


def test_schema_tenure_string_invalida(dados_cliente):
    dados_cliente["tenure"] = "Banana"
    with pytest.raises(ValidationError) as exc_info:
        ClienteChurn.model_validate(dados_cliente)

    erros = exc_info.value.errors()
    assert any(
        erro["loc"] == ("tenure",) and erro["type"] == "int_parsing"
        for erro in erros
    )


@pytest.mark.parametrize("campo_numerico", ["tenure", "MonthlyCharges", "TotalCharges"])
def test_schema_valores_numericos_negativos_rejeitados(dados_cliente, campo_numerico):
    dados_cliente[campo_numerico] = -1
    with pytest.raises(ValidationError) as exc_info:
        ClienteChurn.model_validate(dados_cliente)

    erros = exc_info.value.errors()
    assert any(
        erro["loc"] == (campo_numerico,) and erro["type"] == "greater_than_equal"
        for erro in erros
    )


@pytest.mark.parametrize(
    "campo,valor_invalido",
    [
        ("Contract", "Monthly"),
        ("gender", "Outro"),
        ("InternetService", "5G"),
        ("PaymentMethod", "Pix"),
        ("Partner", "Talvez"),
        ("PaperlessBilling", "Sim"),
    ],
)
def test_schema_categorias_desconhecidas_rejeitadas(
    dados_cliente, campo, valor_invalido
):
    """Regressao da auditoria V1: todas as 15 colunas categoricas usam Literal estrito."""
    dados_cliente[campo] = valor_invalido
    with pytest.raises(ValidationError) as exc_info:
        ClienteChurn.model_validate(dados_cliente)

    erros = exc_info.value.errors()
    assert any(
        erro["loc"] == (campo,) and erro["type"] == "literal_error"
        for erro in erros
    )


def test_schema_lote_vazio_rejeitado():
    with pytest.raises(ValidationError) as exc_info:
        LoteClientesChurn.model_validate({"clientes": []})

    erros = exc_info.value.errors()
    assert any(
        erro["loc"] == ("clientes",) and erro["type"] == "too_short"
        for erro in erros
    )


def test_schema_lote_acima_do_limite_rejeitado(dados_cliente):
    payload_excessivo = {"clientes": [dados_cliente] * (LIMITE_MAXIMO_LOTE + 1)}
    with pytest.raises(ValidationError) as exc_info:
        LoteClientesChurn.model_validate(payload_excessivo)

    erros = exc_info.value.errors()
    assert any(
        erro["loc"] == ("clientes",) and erro["type"] == "too_long"
        for erro in erros
    )
