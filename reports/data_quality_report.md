# Relatório de Qualidade dos Dados — Telco Customer Churn

Relatório gerado automaticamente por `src/data/validation.py` diretamente sobre o arquivo bruto `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (mantido sem alterações).

## 1. Resumo Geral e Integridade

| Verificação | Resultado | Status |
| --- | --- | --- |
| Dimensões | 7,043 linhas × 21 colunas | OK |
| Colunas obrigatórias | Ausentes: 0 · Extras: 0 | OK |
| Valores ausentes explícitos (`NaN`) | 0 | OK |
| Registros duplicados exatos | 0 | OK |
| `customerID` duplicados | 0 | OK |
| `TotalCharges` inválidos ao converter para número | 11 registros (`' '`) | TRATADO NA PIPELINE |
| Categorias inesperadas | 0 colunas afetadas | OK |
| Alinhamento Dataset × Schema da API | 19 features × 19 campos | OK |
| SHA-256 do CSV bruto | `a0ea74c48c1063be89a73ce3fb868593604c0129f2bf581934f04983d728b3b5` | OK |

## 2. Distribuição da Variável Alvo (`Churn`)

| Classe | Quantidade | Percentual |
| --- | ---: | ---: |
| `No` | 5,174 | 73.46% |
| `Yes` | 1,869 | 26.54% |

## 3. Problema Identificado e Solução: `TotalCharges`

- **Problema:** A coluna `TotalCharges` é carregada como texto (`str`) e possui **11** registros com string contendo apenas espaço em branco (`[' ']`).
- **Causa provável no domínio:** Todos os **11** registros inválidos têm `tenure = 0` (distribuição de `tenure` entre os inválidos: `{0: 11}`), igual ao total de clientes com `tenure = 0` na base (**11**). São clientes recém-chegados que ainda não completaram o primeiro ciclo de cobrança acumulada.
- **Solução adotada (sem data leakage):** O CSV bruto permanece intacto. A conversão `pd.to_numeric(..., errors='coerce')` ocorre dentro do `FunctionTransformer` da `Pipeline`, seguida de `SimpleImputer(strategy='median')` ajustado **exclusivamente** no conjunto de treino.

## 4. Tipos das Colunas e Valores Ausentes Explícitos

| Coluna | Tipo lido pelo Pandas | Ausentes explícitos |
| --- | --- | ---: |
| `customerID` | `str` | 0 |
| `gender` | `str` | 0 |
| `SeniorCitizen` | `int64` | 0 |
| `Partner` | `str` | 0 |
| `Dependents` | `str` | 0 |
| `tenure` | `int64` | 0 |
| `PhoneService` | `str` | 0 |
| `MultipleLines` | `str` | 0 |
| `InternetService` | `str` | 0 |
| `OnlineSecurity` | `str` | 0 |
| `OnlineBackup` | `str` | 0 |
| `DeviceProtection` | `str` | 0 |
| `TechSupport` | `str` | 0 |
| `StreamingTV` | `str` | 0 |
| `StreamingMovies` | `str` | 0 |
| `Contract` | `str` | 0 |
| `PaperlessBilling` | `str` | 0 |
| `PaymentMethod` | `str` | 0 |
| `MonthlyCharges` | `float64` | 0 |
| `TotalCharges` | `str` | 0 |
| `Churn` | `str` | 0 |

## 5. Validação de Faixas Numéricas e Estatísticas Descritivas

| Variável | Mínimo | Mediana | Média | Máximo |
| --- | ---: | ---: | ---: | ---: |
| `tenure` | 0.00 | 29.00 | 32.37 | 72.00 |
| `MonthlyCharges` | 18.25 | 70.35 | 64.76 | 118.75 |
| `TotalCharges_convertido` | 18.80 | 1397.47 | 2283.30 | 8684.80 |

**Checagens de valores numéricos inválidos:**
- `tenure < 0`: **0**
- `MonthlyCharges < 0`: **0**
- `TotalCharges < 0`: **0**
- `SeniorCitizen` fora de `{0, 1}`: **0**

## 6. Consistência Lógica entre Serviços

- `PhoneService == 'No'` com `MultipleLines != 'No phone service'`: **0**
- `InternetService == 'No'` com `OnlineSecurity != 'No internet service'`: **0**
- `InternetService == 'No'` com `OnlineBackup != 'No internet service'`: **0**
- `InternetService == 'No'` com `DeviceProtection != 'No internet service'`: **0**
- `InternetService == 'No'` com `TechSupport != 'No internet service'`: **0**
- `InternetService == 'No'` com `StreamingTV != 'No internet service'`: **0**
- `InternetService == 'No'` com `StreamingMovies != 'No internet service'`: **0**

## 7. Consistência com os Schemas da API

- Features de entrada no modelo: **19** (`customerID` e `Churn` removidas).
- Campos definidos no contrato Pydantic (`ClienteChurn`): **19**.
- Campos faltando na API: `[]`.
- Campos extras na API: `[]`.
- Política para categorias desconhecidas: todas as 15 colunas categóricas são validadas estritamente via `Literal[...]` no Pydantic (retornando HTTP `422`), enquanto o `OneHotEncoder(handle_unknown='ignore')` permanece na pipeline como defesa em profundidade.

## 8. Redistribuição e Atribuição do Dataset

- **Atribuição oficial:** *IBM Sample Data Sets (Watson Analytics / Telco Customer Churn sample)*, disponibilizado no repositório [IBM/customer-churn-prediction](https://github.com/IBM/customer-churn-prediction).
- **Licenciamento e escopo de redistribuição:** o repositório de código da IBM é licenciado sob **Apache License 2.0**, enquanto o CSV `WA_Fn-UseC_-Telco-Customer-Churn.csv` é um dataset fictício de demonstração publicado pela IBM para fins educacionais e analíticos (sem uma licença aberta dedicada que autorize redistribuição comercial irrestrita do CSV bruto).
- **Preservação neste portfólio:** o arquivo já integrava a versão V1 deste repositório educacional e é mantido sem alterações nos registros, com quebras de linha canônicas `LF` protegidas por `.gitattributes` e atribuição explícita à IBM em `data/README.md`.
