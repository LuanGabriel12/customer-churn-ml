# Referência da API — Customer Churn Intelligence API (v2.0.0)

Documentação completa dos endpoints, schemas Pydantic, regras de validação, políticas de erro e exemplos reais executados sobre a API local (`uvicorn app.main:app --reload`).

- **Interface interativa (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Documentação alternativa (ReDoc):** `http://127.0.0.1:8000/redoc`
- **Contrato OpenAPI JSON:** `http://127.0.0.1:8000/openapi.json`

---

## 1. Resumo dos Endpoints

| Método | Rota | Finalidade | Códigos HTTP |
| --- | --- | --- | --- |
| `GET` | `/health` | Verifica se a aplicação está ativa e se o modelo está carregado em memória. | `200`, `503` |
| `GET` | `/model-info` | Retorna metadados verificáveis do modelo carregado, features, classes e métricas. | `200`, `503` |
| `POST` | `/predict` | Executa previsão individual de churn (contrato preservado da V1). | `200`, `422`, `503` |
| `POST` | `/predict/batch` | Executa previsão em lote para 1 até 100 clientes em uma única requisição. | `200`, `422`, `503` |

---

## 2. Política de Validação e Tratamento de Erros

- **19 features obrigatórias:** todos os campos de `ClienteChurn` devem estar presentes e correspondem exatamente às 19 colunas usadas no treinamento (`src/config.py`).
- **Tipos e limites numéricos:**
  - `tenure`: `int >= 0`
  - `MonthlyCharges`: `float >= 0` (rejeita `NaN` e `Inf`)
  - `TotalCharges`: `float >= 0` (rejeita `NaN` e `Inf`)
  - `SeniorCitizen`: `int` com `0 <= SeniorCitizen <= 1`
- **Política para categorias desconhecidas:** todas as 15 colunas categóricas de texto usam `Literal[...]` estrito no Pydantic. Qualquer valor fora das categorias válidas do dataset (por exemplo, `"Contract": "Monthly"` ou `"PaymentMethod": "Pix"`) retorna **HTTP 422 Unprocessable Entity** com a descrição exata do campo inválido. Como defesa em profundidade, o `OneHotEncoder(handle_unknown="ignore")` permanece na pipeline de ML.
- **Proteções do lote (`POST /predict/batch`):**
  - Lista vazia (`{"clientes": []}`) -> **HTTP 422** (`too_short`).
  - Mais de `100` clientes por requisição -> **HTTP 422** (`too_long`).
- **Indisponibilidade do modelo:** se `models/churn_pipeline.joblib` ou `models/model_metadata.json` não estiver disponível — ou se houver divergência entre o modelo carregado e seus metadados (`modelo_sha256`, etapas, classes ou features) — os endpoints respondem com **HTTP 503 Service Unavailable** utilizando mensagem genérica, **sem expor caminhos internos do sistema de arquivos**.
- **Segurança de artefatos `.joblib`:** o módulo `joblib` utiliza serialização `pickle` internamente, que pode executar código arbitrário ao desserializar arquivos maliciosos. Carregue exclusivamente arquivos `.joblib` gerados localmente por `python -m src.modeling.train` ou provenientes de fontes explicitamente confiáveis.
- **Identificação segura da classe positiva:** a API nunca pressupõe que `"Yes"` ocupa o índice `1` em `predict_proba`; os índices são consultados dinamicamente em `modelo.classes_`.

---

## 3. Detalhamento e Exemplos Reais

### 3.1 `GET /health`

**Requisição:**

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Resposta real (`200 OK`):**

```json
{
  "status": "ok",
  "modelo_carregado": true,
  "detalhe": "Aplicacao e modelo prontos para previsao."
}
```

**Quando o modelo está indisponível (`503 Service Unavailable`):**

```json
{
  "detail": "Aplicacao ativa, mas o modelo nao esta carregado. Execute 'python -m src.modeling.train' para gerar o artefato."
}
```

---

### 3.2 `GET /model-info`

**Requisição:**

```bash
curl -X GET "http://127.0.0.1:8000/model-info"
```

**Resposta real (`200 OK`):**

```json
{
  "nome_modelo": "Logistic Regression",
  "algoritmo": "LogisticRegression",
  "versao_projeto": "2.0.0",
  "treinado_em_utc": "2026-11-26T03:01:11Z",
  "etapas_pipeline": [
    "corrigir_total_charges",
    "preprocessador",
    "modelo"
  ],
  "classes_modelo": ["No", "Yes"],
  "classe_positiva": "Yes",
  "quantidade_features": 19,
  "features_esperadas": [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
  ],
  "registros_treino": 5634,
  "registros_teste": 1409,
  "dataset_sha256": "a0ea74c48c1063be89a73ce3fb868593604c0129f2bf581934f04983d728b3b5",
  "modelo_sha256": "<sha256_do_arquivo_models_churn_pipeline_joblib>",
  "limite_maximo_lote": 100,
  "politica_categorias_desconhecidas": "Validacao estrita com Literal no schema Pydantic da API (HTTP 422); OneHotEncoder(handle_unknown='ignore') mantido na pipeline como defesa em profundidade.",
  "metricas_validacao_cruzada": {
    "accuracy_treino": 0.805599836909518,
    "accuracy_validacao": 0.8042220579636595,
    "recall_validacao": 0.5484949832775919,
    "precision_validacao": 0.6576896316176918,
    "f1_validacao": 0.5979882242724168
  },
  "metricas_teste_historico": {
    "accuracy": 0.8055358410220014,
    "recall": 0.5588235294117647,
    "precision": 0.6572327044025157,
    "f1": 0.6040462427745664,
    "matriz_confusao": {
      "TN": 926,
      "FP": 109,
      "FN": 165,
      "TP": 209
    }
  }
}
```

---

### 3.3 `POST /predict`

Preserva exatamente o contrato da API original (`previsao`, `probabilidade_churn`, `probabilidade_nao_churn`).

**Requisição:**

```json
{
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
  "PaymentMethod": "Electronic check"
}
```

**Resposta real (`200 OK`):**

```json
{
  "previsao": "Yes",
  "probabilidade_churn": 0.5945508161523542,
  "probabilidade_nao_churn": 0.40544918384764583
}
```

> Note que `0.5945508161523542 + 0.40544918384764583 = 1.0`. O valor `0.5946` é uma **probabilidade estimada pelo modelo**, não garantia de cancelamento.

---

### 3.4 `POST /predict/batch`

**Requisição (2 clientes: perfil de maior risco e perfil de baixo risco):**

```json
{
  "clientes": [
    {
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
      "PaymentMethod": "Electronic check"
    },
    {
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
      "PaymentMethod": "Credit card (automatic)"
    }
  ]
}
```

**Resposta real (`200 OK`):**

```json
{
  "total_clientes": 2,
  "previsoes": [
    {
      "previsao": "Yes",
      "probabilidade_churn": 0.5945508161523542,
      "probabilidade_nao_churn": 0.40544918384764583
    },
    {
      "previsao": "No",
      "probabilidade_churn": 0.0029748883561160184,
      "probabilidade_nao_churn": 0.997025111643884
    }
  ]
}
```

---

### 3.5 Exemplo de Entrada Inválida (`422 Unprocessable Entity`)

Enviando `"PaymentMethod": "Bitcoin"` e `"tenure": -2` para `POST /predict`:

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "tenure"],
      "msg": "Input should be greater than or equal to 0",
      "input": -2,
      "ctx": {"ge": 0}
    },
    {
      "type": "literal_error",
      "loc": ["body", "PaymentMethod"],
      "msg": "Input should be 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check' or 'Mailed check'",
      "input": "Bitcoin"
    }
  ]
}
```
