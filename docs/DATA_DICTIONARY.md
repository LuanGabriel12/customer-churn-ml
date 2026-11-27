# Dicionário de Dados e Qualidade — Telco Customer Churn

Este documento cataloga todas as 21 colunas do dataset **IBM Telco Customer Churn**, seu tratamento na pipeline de Machine Learning, seu mapeamento para os contratos Pydantic da API e os achados auditados de qualidade.

---

## 1. Origem, Integridade e Atribuição

- **Arquivo bruto:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (obtido e normalizado localmente conforme `data/README.md`).
- **Atribuição oficial:** *IBM Sample Data Sets (Watson Analytics / Telco Customer Churn sample)*, via [IBM/customer-churn-prediction](https://github.com/IBM/customer-churn-prediction).
- **Condições de uso e redistribuição:** O repositório de código da IBM é licenciado sob **Apache License 2.0**, enquanto o CSV bruto é um conjunto de dados **fictício** de amostra publicado pela IBM para fins educacionais e demonstrativos (sem uma licença aberta dedicada que autorize redistribuição comercial irrestrita do arquivo bruto). O CSV não integra a árvore atual; informações de obtenção e atribuição estão em `data/README.md`.
- **Dimensões:** `7.043` registros × `21` colunas.
- **Formato canônico e SHA-256 verificado:** quebras de linha `LF` protegidas via `.gitattributes` (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv -text`) para preservar o hash `a0ea74c48c1063be89a73ce3fb868593604c0129f2bf581934f04983d728b3b5` entre Windows, Linux, Git e arquivos `.zip`.

---

## 2. Catálogo Completo das 21 Colunas

| # | Coluna | Papel no ML | Tipo Bruto (CSV) | Tipo na API (`ClienteChurn`) | Domínio / Valores Observados | Tratamento na Pipeline |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `customerID` | Identificador (removida do X) | `str` | Não enviada na API | `7.043` códigos únicos (ex.: `7590-VHVEG`) | Removida antes do treino; mantida apenas para rastreamento na fila de risco (`risk_ranking.py`). |
| 2 | `gender` | Feature binária | `str` | `Literal["Female", "Male"]` | `Female`, `Male` | `OneHotEncoder(handle_unknown="ignore")` |
| 3 | `SeniorCitizen` | Feature binária | `int64` | `int` (`ge=0, le=1`) | `0` (não idoso), `1` (idoso) | `OneHotEncoder(handle_unknown="ignore")` (preservado da metodologia V1) |
| 4 | `Partner` | Feature binária | `str` | `Literal["No", "Yes"]` | `No`, `Yes` | `OneHotEncoder(handle_unknown="ignore")` |
| 5 | `Dependents` | Feature binária | `str` | `Literal["No", "Yes"]` | `No`, `Yes` | `OneHotEncoder(handle_unknown="ignore")` |
| 6 | `tenure` | Feature numérica | `int64` | `int` (`ge=0`) | `0` a `72` meses (mediana `29`, média `32,37`) | `SimpleImputer(strategy="median")` + `StandardScaler` |
| 7 | `PhoneService` | Feature binária | `str` | `Literal["No", "Yes"]` | `No`, `Yes` | `OneHotEncoder(handle_unknown="ignore")` |
| 8 | `MultipleLines` | Feature categórica | `str` | `Literal["No", "No phone service", "Yes"]` | 3 categorias válidas | `OneHotEncoder(handle_unknown="ignore")` |
| 9 | `InternetService` | Feature categórica | `str` | `Literal["DSL", "Fiber optic", "No"]` | 3 categorias válidas | `OneHotEncoder(handle_unknown="ignore")` |
| 10 | `OnlineSecurity` | Feature categórica | `str` | `Literal["No", "No internet service", "Yes"]` | 3 categorias válidas | `OneHotEncoder(handle_unknown="ignore")` |
| 11 | `OnlineBackup` | Feature categórica | `str` | `Literal["No", "No internet service", "Yes"]` | 3 categorias válidas | `OneHotEncoder(handle_unknown="ignore")` |
| 12 | `DeviceProtection` | Feature categórica | `str` | `Literal["No", "No internet service", "Yes"]` | 3 categorias válidas | `OneHotEncoder(handle_unknown="ignore")` |
| 13 | `TechSupport` | Feature categórica | `str` | `Literal["No", "No internet service", "Yes"]` | 3 categorias válidas | `OneHotEncoder(handle_unknown="ignore")` |
| 14 | `StreamingTV` | Feature categórica | `str` | `Literal["No", "No internet service", "Yes"]` | 3 categorias válidas | `OneHotEncoder(handle_unknown="ignore")` |
| 15 | `StreamingMovies` | Feature categórica | `str` | `Literal["No", "No internet service", "Yes"]` | 3 categorias válidas | `OneHotEncoder(handle_unknown="ignore")` |
| 16 | `Contract` | Feature categórica | `str` | `Literal["Month-to-month", "One year", "Two year"]` | 3 modalidades contratuais | `OneHotEncoder(handle_unknown="ignore")` |
| 17 | `PaperlessBilling` | Feature binária | `str` | `Literal["No", "Yes"]` | `No`, `Yes` | `OneHotEncoder(handle_unknown="ignore")` |
| 18 | `PaymentMethod` | Feature categórica | `str` | `Literal[...]` (4 opções) | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` | `OneHotEncoder(handle_unknown="ignore")` |
| 19 | `MonthlyCharges` | Feature numérica | `float64` | `float` (`ge=0`, sem NaN/Inf) | `18,25` a `118,75` (mediana `70,35`, média `64,76`) | `SimpleImputer(strategy="median")` + `StandardScaler` |
| 20 | `TotalCharges` | Feature numérica (lida como texto no CSV) | `str` | `float` (`ge=0`, sem NaN/Inf) | `18,80` a `8.684,80` + `11` valores `" "` | `FunctionTransformer(corrigir_total_charges)` -> `SimpleImputer(strategy="median")` -> `StandardScaler` |
| 21 | `Churn` | Variável alvo (`y`) | `str` | Saída (`previsao`) | `No` (`5.174` / `73,46%`), `Yes` (`1.869` / `26,54%`) | Classe positiva explícita: `"Yes"` |

---

## 3. Problemas de Qualidade Auditados e Soluções Adotadas

Todos os achados abaixo são verificados automaticamente por `python -m src.data.validation` e registrados em [`reports/data_quality_report.md`](../reports/data_quality_report.md):

1. **`TotalCharges` armazenada como texto com 11 strings em branco (`" "`):**
   - **Diagnóstico:** Na leitura do CSV bruto, `TotalCharges.isna().sum()` retorna `0` porque espaços em branco não são reconhecidos como `NaN` pelo `pd.read_csv`. Ao aplicar `pd.to_numeric(..., errors="coerce")`, surgem exatamente **11 valores ausentes**.
   - **Relação de domínio:** Todos os **11** registros com `TotalCharges = " "` têm `tenure == 0` — e existem exatamente **11** clientes com `tenure == 0` na base inteira. Trata-se de clientes recém-ativados sem faturamento acumulado fechado.
   - **Solução adotada:** Preservar o CSV original intacto e executar a conversão dentro da `Pipeline` via `FunctionTransformer(corrigir_total_charges)`, seguido de `SimpleImputer(strategy="median")` ajustado **somente no treino** (`mediana de treino = 1.398,125`, diferente da mediana da base inteira `1.397,475`). Isso preserva 100% a metodologia V1 e elimina data leakage.
   - **Nota sobre a API:** Na API, `ClienteChurn` exige `TotalCharges: float = Field(ge=0)` (permitindo `0.0` para clientes novos com `tenure = 0`), mantendo compatibilidade com os testes originais da API. Caso a pipeline seja chamada diretamente via Python com `" "`, o `FunctionTransformer` trata o caso sem erro.

2. **Ausência de duplicidades e chaves íntegras:**
   - Linhas exatamente duplicadas: `0`.
   - `customerID` duplicados: `0`.
   - Valores numéricos negativos (`tenure`, `MonthlyCharges`, `TotalCharges`): `0`.
   - `SeniorCitizen` fora de `{0, 1}`: `0`.

3. **Consistência lógica entre serviços:**
   - Clientes com `PhoneService == "No"` e `MultipleLines != "No phone service"`: `0`.
   - Clientes com `InternetService == "No"` e qualquer serviço adicional de internet diferente de `"No internet service"`: `0`.

4. **Desbalanceamento moderado da variável alvo (`Churn`):**
   - `No`: `5.174` (`73,46%`) vs. `Yes`: `1.869` (`26,54%`).
   - **Solução:** Divisão estratificada (`stratify=y`) e avaliação obrigatória com Recall, Precision, F1-score, Matriz de Confusão e comparação contra a baseline `DummyClassifier(strategy="most_frequent")`.

5. **Alinhamento Dataset × Schema da API:**
   - Todas as `19` features do modelo coincidem exatamente com os `19` campos de `ClienteChurn`.
   - Na API original V1, apenas `Contract` validava categorias com `Literal`; na V2, todas as `15` colunas categóricas de texto usam `Literal[...]` alinhados às categorias auditadas acima, rejeitando valores desconhecidos com HTTP `422`.

---

## 4. Colunas Derivadas da Camada Analítica (Não Usadas no Modelo)

As colunas abaixo existem **exclusivamente** em memória durante a execução de `src/analytics/` para gerar relatórios comerciais e fila de priorização, sem alterar o treinamento do modelo:

- `faixa_tenure`: `01. 0 a 12 meses`, `02. 13 a 24 meses`, `03. 25 a 48 meses`, `04. 49 a 72 meses`.
- `faixa_cobranca_mensal`: `01. Ate US$ 35,00`, `02. US$ 35,01 a US$ 70,00`, `03. US$ 70,01 a US$ 90,00`, `04. Acima de US$ 90,00`.
- `contrato_x_internet`: concatenação legível de `Contract` + `InternetService`.
- `faixa_risco`: agrupamento operacional da probabilidade estimada em `Alto (>= 0,60)`, `Moderado (0,35 a 0,5999)` e `Baixo (< 0,35)`.
