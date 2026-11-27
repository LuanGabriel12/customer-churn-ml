"""Fixtures compartilhadas para os testes de schema, modelagem e API."""

from fastapi.testclient import TestClient
import pytest

from app.main import ESTADO_APLICACAO, app, inicializar_modelo


@pytest.fixture
def dados_cliente():
    """Retorna um payload valido de cliente com todas as 19 features esperadas."""
    return {
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }


@pytest.fixture
def dados_cliente_baixo_risco():
    """Retorna um segundo payload valido (perfil de contrato longo e baixo risco)."""
    return {
        "tenure": 60,
        "MonthlyCharges": 25.0,
        "TotalCharges": 1500.0,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
    }


@pytest.fixture
def client():
    """Garante o modelo padrao carregado e retorna o cliente de testes FastAPI."""
    if ESTADO_APLICACAO["modelo"] is None:
        inicializar_modelo()
    return TestClient(app)


@pytest.fixture
def api_sem_modelo():
    """Simula indisponibilidade do modelo durante um teste e restaura o estado ao final."""
    estado_anterior = dict(ESTADO_APLICACAO)
    ESTADO_APLICACAO["modelo"] = None
    ESTADO_APLICACAO["metadados"] = None
    ESTADO_APLICACAO["erro_carregamento"] = "Modelo simulado como indisponivel no teste."
    try:
        yield TestClient(app)
    finally:
        ESTADO_APLICACAO.update(estado_anterior)
