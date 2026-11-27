# Metodologia de Machine Learning e Avaliação

Este documento registra com rigor metodológico o desenho experimental, o controle contra *data leakage*, a reprodução da baseline histórica V1, a comparação contra `DummyClassifier` e as limitações conhecidas.

---

## 1. Enquadramento do Problema

- **Tipo de problema:** Classificação supervisionada binária.
- **Variável alvo (`y`):** `Churn` (`"Yes"` = classe positiva; `"No"` = classe negativa).
- **Pergunta de negócio:** Dado o perfil contratual, de serviços e de cobrança de um cliente, qual é a probabilidade estimada de cancelamento e como priorizar clientes em uma fila de risco?
- **Restrição epistemológica:** O modelo identifica **padrões estatísticos associados** ao cancelamento no histórico; ele **não** estabelece relações causais e **não** garante que ações de retenção gerem lucro sem testes experimentais (como testes A/B em campo).

---

## 2. Separação dos Dados e Prevenção de Data Leakage

- **Remoção de identificador:** `customerID` é descartada antes do split por identificar o registro individual sem representar comportamento generalizável.
- **Divisão preservada da V1:** `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`.
  - **Treino:** `5.634` clientes (`4.139` `No` / `1.495` `Yes` — `26,54%` churn).
  - **Teste:** `1.409` clientes (`1.035` `No` / `374` `Yes` — `26,54%` churn).
- **Por que não há data leakage no pré-processamento:**
  1. A mediana do `SimpleImputer(strategy="median")` é aprendida exclusivamente sobre `X_train` (`TotalCharges = 1.398,125`, verificado por teste automatizado contra a mediana da base inteira `1.397,475`).
  2. Média e desvio-padrão do `StandardScaler` e o vocabulário do `OneHotEncoder` são ajustados apenas dentro do treino (e separadamente em cada fold da validação cruzada).
  3. A validação cruzada (`cv=5`, `return_train_score=True`) roda **somente** sobre `(X_train, y_train)`.

> **Transparência sobre o conjunto de teste histórico:** O split `random_state=42, test_size=0.20` já havia sido avaliado na V1. Por disciplina metodológica, este conjunto de teste **não** é tratado como uma avaliação inédita e **não** foi utilizado para tunar hiperparâmetros, trocar de algoritmo ou otimizar threshold na V2.

---

## 3. Pipeline de Pré-processamento

Implementada em `src/modeling/preprocessing.py` (preservando integralmente a definição original da V1):

1. **`corrigir_total_charges` (`FunctionTransformer`):** copia o `DataFrame` e converte `TotalCharges` com `pd.to_numeric(..., errors="coerce")`, transformando strings inválidas (`" "`) em `NaN`.
2. **`preprocessador` (`ColumnTransformer`):**
   - **Numéricas** (`tenure`, `MonthlyCharges`, `TotalCharges`): `SimpleImputer(strategy="median")` -> `StandardScaler()`.
   - **Categóricas e binárias** (16 colunas, incluindo `SeniorCitizen` conforme a V1): `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
3. **`modelo`:** estimador supervisionado (`DummyClassifier`, `LogisticRegression` ou `DecisionTreeClassifier`).

---

## 4. Modelos Comparados e Resultados Reproduzidos

Todos os números abaixo foram gerados de forma reproduzível por `python -m src.modeling.train` (exportados em [`reports/model_comparison.csv`](../reports/model_comparison.csv) e [`reports/model_metrics.json`](../reports/model_metrics.json)):

### 4.1 Validação Cruzada (5-fold exclusivamente no Treino — 5.634 clientes)

| Modelo | Accuracy Treino (CV) | Accuracy Validação (CV) | Recall Validação | Precision Validação | F1-score Validação | Diagnóstico |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **DummyClassifier (`most_frequent`)** | 73,46% | 73,46% | 0,00% | 0,00% | 0,00% | Baseline ingênua (sempre prevê `"No"`) |
| **Logistic Regression** | 80,56% | **80,42%** | **54,85%** | **65,77%** | **59,80%** | Melhor equilíbrio e ausência de overfitting |
| **Decision Tree (sem limite)** | 99,84% | 72,01% | 48,49% | 47,37% | 47,89% | Forte overfitting (gap de 27,83 p.p. e accuracy inferior à baseline) |
| **Decision Tree (`max_depth=4`)** | 79,80% | 78,72% | 47,76% | 64,23% | 54,20% | Controlou o overfitting, mas ficou abaixo da Logistic Regression |

### 4.2 Avaliação no Conjunto de Teste Histórico (1.409 clientes)

| Modelo | Accuracy | Recall | Precision | F1-score | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DummyClassifier (`most_frequent`)** | 73,46% | 0,00% | 0,00% | 0,00% | 1.035 | 0 | 374 | 0 |
| **Logistic Regression (Modelo Final)** | **80,55%** | **55,88%** | 65,72% | **60,40%** | 926 | 109 | 165 | 209 |
| **Decision Tree (sem limite)** | 72,75% | 50,53% | 48,71% | 49,61% | 836 | 199 | 185 | 189 |
| **Decision Tree (`max_depth=4`)** | 79,21% | 45,19% | **65,76%** | 53,57% | 947 | 88 | 205 | 169 |

### 4.3 Verificação de Reprodução vs. Referência Histórica V1

| Métrica (`Logistic Regression` no Teste) | Referência Histórica V1 | Reprodução V2 | Divergência |
| --- | ---: | ---: | ---: |
| Accuracy | 80,55% | 80,55% (`0,805536`) | 0,00 p.p. |
| Recall | 55,88% | 55,88% (`0,558824`) | 0,00 p.p. |
| Precision | 65,72% | 65,72% (`0,657233`) | 0,00 p.p. |
| F1-score | 60,40% | 60,40% (`0,604046`) | 0,00 p.p. |
| Matriz de Confusão (`TN / FP / FN / TP`) | `926 / 109 / 165 / 209` | `926 / 109 / 165 / 209` | 0 casos |

---

## 5. Interpretação de Negócio: Erros e Superação da Baseline

1. **O modelo supera uma baseline simples?**
   - **Sim.** O `DummyClassifier` alcança `73,46%` de accuracy apenas prevendo `"No"` para todos os clientes, mas falha completamente no objetivo de negócio (`Recall = 0%`, perdendo todos os `374` clientes que cancelaram). A `Logistic Regression` eleva a accuracy para `80,55%` (`+7,09 p.p.`), o Recall para `55,88%` (`209` cancelamentos identificados) e o F1-score de `0,00%` para `60,40%`.
2. **Consequências dos Falsos Positivos (`FP = 109`):**
   - Clientes que não cancelariam, mas foram sinalizados como risco de churn.
   - **Impacto operacional:** consumo de tempo da equipe de retenção, possível contato desnecessário e custo de eventuais ofertas/descontos concedidos a quem permaneceria na base.
3. **Consequências dos Falsos Negativos (`FN = 165`):**
   - Clientes que realmente cancelaram, mas passaram abaixo do corte padrão (`0,50`) e foram classificados como `"No"`.
   - **Impacto operacional:** representam a principal limitação atual do modelo, pois são oportunidades de atuação preventiva que deixam de ser sinalizadas na classificação binária.
4. **Uso do score contínuo (`predict_proba`) para priorização:**
   - Mesmo mantendo o threshold padrão `0,50` para a previsão binária (`Yes`/`No`), ordenar clientes pela probabilidade estimada permite priorizar quem tem maior risco primeiro. Na amostra de teste (`1.409` clientes), a faixa operacional **Alto (`>= 0,60`)** concentra `209` clientes com taxa real de churn de **71,77%**, enquanto a faixa **Baixo (`< 0,35`)** tem `924` clientes com taxa real de **11,90%**.

---

## 6. Limitações Metodológicas Declaradas

- **Recall limitado (`55,88%`):** `165` dos `374` cancelamentos no teste não são classificados como `"Yes"` no threshold padrão `0,50`.
- **Conjunto de teste único e histórico:** já observado na V1; uma evolução futura exigiria validação temporal ou novos dados coletados em campo.
- **Ausência de matriz de custos real:** não existem dados reais de custo de contato, taxa de sucesso de campanhas de retenção ou LTV líquido por cliente; portanto, não se faz qualquer alegação de ganho financeiro ou ROI.
- **Dataset demonstrativo:** a base da IBM não inclui histórico de tickets de suporte, qualidade de rede, quedas de conexão ou sazonalidade mensal.
- **Projeto fora de produção:** trata-se de uma arquitetura de portfólio para demonstração rigorosa de engenharia de ML e APIs.
