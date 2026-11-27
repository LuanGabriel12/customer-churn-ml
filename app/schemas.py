"""Schemas Pydantic de entrada e saida da API de previsao de churn."""

from typing import Literal

from pydantic import BaseModel, Field

from src.config import LIMITE_MAXIMO_LOTE


class ClienteChurn(BaseModel):
    """Contrato das 19 variaveis de entrada alinhadas ao treinamento do modelo."""

    tenure: int = Field(ge=0, description="Tempo de permanencia como cliente, em meses.")
    MonthlyCharges: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Valor da cobranca mensal atual (US$).",
    )
    TotalCharges: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Valor total acumulado cobrado do cliente (US$).",
    )
    gender: Literal["Female", "Male"] = Field(description="Genero registrado do cliente.")
    SeniorCitizen: int = Field(
        ge=0,
        le=1,
        description="Indicador binario de cliente idoso (0 = Nao, 1 = Sim).",
    )
    Partner: Literal["No", "Yes"] = Field(description="Indica se possui parceiro(a).")
    Dependents: Literal["No", "Yes"] = Field(description="Indica se possui dependentes.")
    PhoneService: Literal["No", "Yes"] = Field(
        description="Indica se possui servico telefonico."
    )
    MultipleLines: Literal["No", "No phone service", "Yes"] = Field(
        description="Indica se possui multiplas linhas telefonicas."
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        description="Tipo de servico de internet contratado."
    )
    OnlineSecurity: Literal["No", "No internet service", "Yes"] = Field(
        description="Servico adicional de seguranca online."
    )
    OnlineBackup: Literal["No", "No internet service", "Yes"] = Field(
        description="Servico adicional de backup online."
    )
    DeviceProtection: Literal["No", "No internet service", "Yes"] = Field(
        description="Servico adicional de protecao de dispositivo."
    )
    TechSupport: Literal["No", "No internet service", "Yes"] = Field(
        description="Servico adicional de suporte tecnico."
    )
    StreamingTV: Literal["No", "No internet service", "Yes"] = Field(
        description="Servico adicional de streaming de TV."
    )
    StreamingMovies: Literal["No", "No internet service", "Yes"] = Field(
        description="Servico adicional de streaming de filmes."
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="Modalidade de contrato vigente."
    )
    PaperlessBilling: Literal["No", "Yes"] = Field(
        description="Indica se a fatura e digital (sem papel)."
    )
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(description="Forma de pagamento utilizada.")


class RespostaChurn(BaseModel):
    """Contrato preservado de resposta para previsao individual."""

    previsao: str = Field(description="Classe prevista pelo modelo ('Yes' ou 'No').")
    probabilidade_churn: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada para a classe positiva ('Yes').",
    )
    probabilidade_nao_churn: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada para a classe negativa ('No').",
    )


class LoteClientesChurn(BaseModel):
    """Contrato de entrada para previsoes em lote com protecao contra lista vazia e excesso."""

    clientes: list[ClienteChurn] = Field(
        min_length=1,
        max_length=LIMITE_MAXIMO_LOTE,
        description=f"Lista de clientes para previsao (minimo 1, maximo {LIMITE_MAXIMO_LOTE}).",
    )


class RespostaLoteChurn(BaseModel):
    """Resposta consolidada para previsao em lote."""

    total_clientes: int = Field(ge=1, description="Quantidade de clientes processados.")
    previsoes: list[RespostaChurn] = Field(
        description="Lista de previsoes na mesma ordem dos clientes enviados."
    )


class RespostaSaude(BaseModel):
    """Status de disponibilidade da aplicacao e do artefato do modelo."""

    status: str
    modelo_carregado: bool
    detalhe: str


class InfoModelo(BaseModel):
    """Informacoes verificaveis sobre o modelo carregado em memoria."""

    nome_modelo: str
    algoritmo: str
    versao_projeto: str
    treinado_em_utc: str
    etapas_pipeline: list[str]
    classes_modelo: list[str]
    classe_positiva: str
    quantidade_features: int
    features_esperadas: list[str]
    registros_treino: int
    registros_teste: int
    dataset_sha256: str
    modelo_sha256: str
    limite_maximo_lote: int
    politica_categorias_desconhecidas: str
    metricas_validacao_cruzada: dict
    metricas_teste_historico: dict
