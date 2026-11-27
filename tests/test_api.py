"""Testes de integracao da FastAPI: /health, /model-info, /predict, /predict/batch e erros."""

import json

from fastapi.testclient import TestClient
import pytest

from app.main import ESTADO_APLICACAO, app, inicializar_modelo
from src.config import ARQUIVO_METADADOS_MODELO, ARQUIVO_MODELO, LIMITE_MAXIMO_LOTE
from src.modeling.predict import prever_cliente, prever_lote


def test_health_ok(client):
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["modelo_carregado"] is True


def test_model_info_retorna_metadados_verificaveis(client):
    resposta = client.get("/model-info")
    assert resposta.status_code == 200
    corpo = resposta.json()

    assert corpo["nome_modelo"] == "Logistic Regression"
    assert corpo["algoritmo"] == "LogisticRegression"
    assert corpo["classe_positiva"] == "Yes"
    assert set(corpo["classes_modelo"]) == {"No", "Yes"}
    assert corpo["quantidade_features"] == 19
    assert corpo["registros_treino"] == 5634
    assert corpo["registros_teste"] == 1409
    assert len(corpo["modelo_sha256"]) == 64
    assert corpo["limite_maximo_lote"] == LIMITE_MAXIMO_LOTE
    assert "corrigir_total_charges" in corpo["etapas_pipeline"]
    assert corpo["metricas_teste_historico"]["matriz_confusao"] == {
        "TN": 926,
        "FP": 109,
        "FN": 165,
        "TP": 209,
    }


def test_predict_valido_e_consistente_com_previsao_direta(dados_cliente, client):
    resposta = client.post("/predict", json=dados_cliente)
    assert resposta.status_code == 200

    corpo = resposta.json()
    assert set(corpo.keys()) == {
        "previsao",
        "probabilidade_churn",
        "probabilidade_nao_churn",
    }
    assert corpo["previsao"] in ["Yes", "No"]

    prob_yes = corpo["probabilidade_churn"]
    prob_no = corpo["probabilidade_nao_churn"]
    assert 0.0 <= prob_yes <= 1.0
    assert 0.0 <= prob_no <= 1.0
    assert (prob_yes + prob_no) == pytest.approx(1.0)

    if prob_yes > prob_no:
        assert corpo["previsao"] == "Yes"
    if prob_yes < prob_no:
        assert corpo["previsao"] == "No"

    # Consistencia direta entre API e modulo src.modeling.predict
    direto = prever_cliente(ESTADO_APLICACAO["modelo"], dados_cliente)
    assert corpo["previsao"] == direto["previsao"]
    assert prob_yes == pytest.approx(direto["probabilidade_churn"])
    assert prob_no == pytest.approx(direto["probabilidade_nao_churn"])


def test_predict_tenure_zero(dados_cliente, client):
    dados_cliente["tenure"] = 0
    dados_cliente["TotalCharges"] = 0.0

    resposta = client.post("/predict", json=dados_cliente)
    assert resposta.status_code == 200
    assert resposta.json()["previsao"] in ["Yes", "No"]


def test_predict_batch_valido_e_consistente(
    dados_cliente, dados_cliente_baixo_risco, client
):
    payload = {"clientes": [dados_cliente, dados_cliente_baixo_risco]}
    resposta = client.post("/predict/batch", json=payload)
    assert resposta.status_code == 200

    corpo = resposta.json()
    assert corpo["total_clientes"] == 2
    assert len(corpo["previsoes"]) == 2

    esperadas = prever_lote(
        ESTADO_APLICACAO["modelo"], [dados_cliente, dados_cliente_baixo_risco]
    )
    for indice in range(2):
        obtida = corpo["previsoes"][indice]
        esperada = esperadas[indice]
        assert obtida["previsao"] == esperada["previsao"]
        assert obtida["probabilidade_churn"] == pytest.approx(
            esperada["probabilidade_churn"]
        )
        assert obtida["probabilidade_nao_churn"] == pytest.approx(
            esperada["probabilidade_nao_churn"]
        )
        assert (
            obtida["probabilidade_churn"] + obtida["probabilidade_nao_churn"]
        ) == pytest.approx(1.0)

    # Perfil de contrato mensal + fibra tende a ter maior risco que contrato 2 anos sem internet
    assert (
        corpo["previsoes"][0]["probabilidade_churn"]
        > corpo["previsoes"][1]["probabilidade_churn"]
    )


def test_predict_batch_lista_vazia_retorna_422(client):
    resposta = client.post("/predict/batch", json={"clientes": []})
    assert resposta.status_code == 422
    assert any(
        erro["loc"] == ["body", "clientes"] and erro["type"] == "too_short"
        for erro in resposta.json()["detail"]
    )


def test_predict_batch_acima_do_limite_retorna_422(dados_cliente, client):
    payload = {"clientes": [dados_cliente] * (LIMITE_MAXIMO_LOTE + 1)}
    resposta = client.post("/predict/batch", json=payload)
    assert resposta.status_code == 422
    assert any(
        erro["loc"] == ["body", "clientes"] and erro["type"] == "too_long"
        for erro in resposta.json()["detail"]
    )


@pytest.mark.parametrize("tenure_invalido", [-1, -2, -100])
def test_predict_tenure_invalido(dados_cliente, tenure_invalido, client):
    dados_cliente["tenure"] = tenure_invalido
    resposta = client.post("/predict", json=dados_cliente)

    assert resposta.status_code == 422
    assert any(
        erro["loc"] == ["body", "tenure"] and erro["type"] == "greater_than_equal"
        for erro in resposta.json()["detail"]
    )


@pytest.mark.parametrize(
    "campo_ausente", ["PaymentMethod", "Contract", "gender", "InternetService"]
)
def test_predict_campo_ausente(dados_cliente, campo_ausente, client):
    dados_cliente.pop(campo_ausente)
    resposta = client.post("/predict", json=dados_cliente)

    assert resposta.status_code == 422
    assert any(
        erro["loc"] == ["body", campo_ausente] and erro["type"] == "missing"
        for erro in resposta.json()["detail"]
    )


@pytest.mark.parametrize("contrato_invalido", ["Banana", "Monthly", ""])
def test_predict_categoria_invalida(dados_cliente, contrato_invalido, client):
    dados_cliente["Contract"] = contrato_invalido
    resposta = client.post("/predict", json=dados_cliente)

    assert resposta.status_code == 422
    assert any(
        erro["loc"] == ["body", "Contract"] and erro["type"] == "literal_error"
        for erro in resposta.json()["detail"]
    )


def test_predict_outra_categoria_desconhecida_retorna_422(dados_cliente, client):
    dados_cliente["PaymentMethod"] = "Bitcoin"
    resposta = client.post("/predict", json=dados_cliente)

    assert resposta.status_code == 422
    assert any(
        erro["loc"] == ["body", "PaymentMethod"] and erro["type"] == "literal_error"
        for erro in resposta.json()["detail"]
    )


def test_predict_dois_campos_ausentes(dados_cliente, client):
    dados_cliente.pop("Contract")
    dados_cliente.pop("PaymentMethod")
    resposta = client.post("/predict", json=dados_cliente)
    erros = resposta.json()["detail"]

    assert resposta.status_code == 422
    assert len(erros) == 2
    assert any(
        erro["loc"] == ["body", "Contract"] and erro["type"] == "missing"
        for erro in erros
    )
    assert any(
        erro["loc"] == ["body", "PaymentMethod"] and erro["type"] == "missing"
        for erro in erros
    )


def test_predict_erros_mistos(dados_cliente, client):
    dados_cliente["tenure"] = -2
    dados_cliente.pop("Contract")
    resposta = client.post("/predict", json=dados_cliente)
    erros = resposta.json()["detail"]

    assert resposta.status_code == 422
    assert len(erros) == 2
    assert any(
        erro["loc"] == ["body", "tenure"] and erro["type"] == "greater_than_equal"
        for erro in erros
    )
    assert any(
        erro["loc"] == ["body", "Contract"] and erro["type"] == "missing"
        for erro in erros
    )


def test_endpoints_retornam_503_quando_modelo_indisponivel(
    dados_cliente, api_sem_modelo
):
    resp_health = api_sem_modelo.get("/health")
    resp_info = api_sem_modelo.get("/model-info")
    resp_predict = api_sem_modelo.post("/predict", json=dados_cliente)
    resp_batch = api_sem_modelo.post("/predict/batch", json={"clientes": [dados_cliente]})

    assert resp_health.status_code == 503
    assert resp_info.status_code == 503
    assert resp_predict.status_code == 503
    assert resp_batch.status_code == 503


def test_inicializar_modelo_caminho_ausente_nao_quebra_aplicacao(tmp_path):
    estado_anterior = dict(ESTADO_APLICACAO)
    try:
        carregado = inicializar_modelo(
            caminho_modelo=tmp_path / "modelo_ausente.joblib",
            caminho_metadados=tmp_path / "metadados_ausentes.json",
        )
        assert carregado is False
        assert ESTADO_APLICACAO["modelo"] is None
        assert ESTADO_APLICACAO["erro_carregamento"] is not None
    finally:
        ESTADO_APLICACAO.update(estado_anterior)


def test_erro_503_nao_expoe_caminhos_internos(dados_cliente, tmp_path):
    """Regressao de seguranca: respostas HTTP 503 nunca devem vazar caminhos do computador."""
    estado_anterior = dict(ESTADO_APLICACAO)
    pasta_interna = tmp_path / "diretorio_interno_privado"
    caminho_inexistente = pasta_interna / "artefato_ausente.joblib"
    try:
        carregado = inicializar_modelo(
            caminho_modelo=caminho_inexistente,
            caminho_metadados=pasta_interna / "metadados_ausentes.json",
        )
        assert carregado is False

        cliente_http = TestClient(app)
        respostas = [
            cliente_http.get("/health"),
            cliente_http.get("/model-info"),
            cliente_http.post("/predict", json=dados_cliente),
            cliente_http.post("/predict/batch", json={"clientes": [dados_cliente]}),
        ]
        for resposta in respostas:
            assert resposta.status_code == 503
            corpo_texto = resposta.text
            assert str(tmp_path) not in corpo_texto
            assert "diretorio_interno_privado" not in corpo_texto
            assert "artefato_ausente.joblib" not in corpo_texto
    finally:
        ESTADO_APLICACAO.update(estado_anterior)


def test_inicializar_modelo_rejeita_metadados_divergentes(tmp_path):
    """Regressao de integridade: recusa carregar se modelo_sha256 divergir dos metadados."""
    estado_anterior = dict(ESTADO_APLICACAO)
    try:
        metadados_adulterados = json.loads(
            ARQUIVO_METADADOS_MODELO.read_text(encoding="utf-8")
        )
        metadados_adulterados["modelo_sha256"] = "0" * 64
        caminho_meta_falso = tmp_path / "model_metadata_adulterado.json"
        caminho_meta_falso.write_text(
            json.dumps(metadados_adulterados), encoding="utf-8"
        )

        carregado = inicializar_modelo(
            caminho_modelo=ARQUIVO_MODELO,
            caminho_metadados=caminho_meta_falso,
        )
        assert carregado is False
        assert ESTADO_APLICACAO["modelo"] is None
    finally:
        ESTADO_APLICACAO.update(estado_anterior)


def test_openapi_contrato_endpoints_v2(client):
    resposta = client.get("/openapi.json")
    assert resposta.status_code == 200
    openapi = resposta.json()
    rotas = openapi["paths"]

    assert "/health" in rotas and "get" in rotas["/health"]
    assert "/model-info" in rotas and "get" in rotas["/model-info"]
    assert "/predict" in rotas and "post" in rotas["/predict"]
    assert "/predict/batch" in rotas and "post" in rotas["/predict/batch"]
    assert "/validar-cliente" not in rotas

    respostas_predict = rotas["/predict"]["post"]["responses"]
    assert "200" in respostas_predict
    assert "422" in respostas_predict

    modelo_resposta = openapi["components"]["schemas"]["RespostaChurn"]
    assert set(modelo_resposta["required"]) == {
        "previsao",
        "probabilidade_churn",
        "probabilidade_nao_churn",
    }
    assert modelo_resposta["properties"]["previsao"]["type"] == "string"
    assert modelo_resposta["properties"]["probabilidade_churn"]["type"] == "number"
    assert modelo_resposta["properties"]["probabilidade_nao_churn"]["type"] == "number"
