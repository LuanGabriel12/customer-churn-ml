"""Configuracoes centralizadas compartilhadas entre dados, modelagem, analises e API."""

from pathlib import Path

# Raiz do repositorio resolvida de forma relativa ao arquivo atual (portavel entre maquinas)
RAIZ_PROJETO = Path(__file__).resolve().parents[1]

# Caminhos principais
DIRETORIO_DADOS = RAIZ_PROJETO / "data"
ARQUIVO_DADOS_BRUTOS = DIRETORIO_DADOS / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

DIRETORIO_MODELOS = RAIZ_PROJETO / "models"
ARQUIVO_MODELO = DIRETORIO_MODELOS / "churn_pipeline.joblib"
ARQUIVO_METADADOS_MODELO = DIRETORIO_MODELOS / "model_metadata.json"

DIRETORIO_RELATORIOS = RAIZ_PROJETO / "reports"
ARQUIVO_RELATORIO_QUALIDADE = DIRETORIO_RELATORIOS / "data_quality_report.md"
ARQUIVO_COMPARACAO_MODELOS = DIRETORIO_RELATORIOS / "model_comparison.csv"
ARQUIVO_METRICAS_MODELOS = DIRETORIO_RELATORIOS / "model_metrics.json"
ARQUIVO_RELATORIO_COMERCIAL = DIRETORIO_RELATORIOS / "commercial_analysis_report.md"
ARQUIVO_CHURN_SEGMENTOS = DIRETORIO_RELATORIOS / "churn_by_segment.csv"
ARQUIVO_FILA_PRIORITARIA = DIRETORIO_RELATORIOS / "priority_customers_test.csv"
ARQUIVO_RELATORIO_RISCO = DIRETORIO_RELATORIOS / "priority_risk_report.md"

# Integridade do dataset original IBM Telco Customer Churn
SHA256_DATASET_ESPERADO = (
    "a0ea74c48c1063be89a73ce3fb868593604c0129f2bf581934f04983d728b3b5"
)

# Variaveis do problema (mesma definicao da metodologia original)
COLUNA_ID = "customerID"
COLUNA_ALVO = "Churn"
CLASSE_POSITIVA = "Yes"
CLASSE_NEGATIVA = "No"
ORDEM_CLASSES_MATRIZ = [CLASSE_NEGATIVA, CLASSE_POSITIVA]

COLUNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

COLUNAS_BINARIAS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

COLUNAS_CATEGORICAS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

COLUNAS_PARA_ONEHOT = COLUNAS_BINARIAS + COLUNAS_CATEGORICAS

# Ordem canonica das 19 features esperadas pelo pipeline (identica ao dataset sem customerID e Churn)
COLUNAS_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

COLUNAS_OBRIGATORIAS_DATASET = [COLUNA_ID] + COLUNAS_FEATURES + [COLUNA_ALVO]

# Categorias validas auditadas no dataset IBM Telco Customer Churn
CATEGORIAS_VALIDAS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "Churn": ["No", "Yes"],
}

# Configuracoes de modelagem (preservadas da V1)
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MAX_ITER_LOGISTICA = 1000
MAX_DEPTH_ARVORE_CONTROLADA = 4
NOME_MODELO_FINAL = "Logistic Regression"

# Limite operacional para requisicoes em lote na API
LIMITE_MAXIMO_LOTE = 100

# Faixas operacionais para fila de priorizacao (nao alteram o threshold 0.50 da classificacao binaria)
LIMITE_RISCO_ALTO = 0.60
LIMITE_RISCO_MODERADO = 0.35
