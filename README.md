# Customer Churn Prediction

## Objetivo

Este projeto busca identificar clientes com risco de cancelamento (*customer churn*) a partir de dados contratuais, demográficos e de serviços de uma empresa de telecomunicações.

## Dataset

Foi utilizado o **Telco Customer Churn**, um conjunto de dados de exemplo publicado pela IBM. Ele contém 7.043 clientes e 21 colunas, incluindo a variável alvo `Churn`.

- Arquivo: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Fonte: [IBM — Telco Customer Churn sample data](https://github.com/IBM/customer-churn-prediction)

O arquivo é mantido sem alterações para que a análise seja reproduzível.

## Problema de negócio

Identificar clientes com maior risco de cancelamento pode ajudar uma empresa a priorizar ações de retenção, como entrar em contato, entender os motivos de insatisfação e avaliar ofertas direcionadas. As métricas deste estudo, por si só, não demonstram impacto financeiro nem garantem que uma ação de retenção será lucrativa.

## Estrutura do projeto

```text
customer-churn-ml/
├── data/
│   ├── README.md
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── .gitignore
├── INTERVIEW_NOTES.md
├── README.md
└── requirements.txt
```

## Preparação dos dados

- `customerID` foi removida por ser apenas um identificador individual.
- Os dados foram divididos em 80% para treino e 20% para teste, com `random_state=42` e `stratify=y`.
- `TotalCharges`, que chega como texto, é convertida para número dentro de um `FunctionTransformer`; valores inválidos passam a ser `NaN`.
- Valores ausentes das variáveis numéricas são preenchidos pela mediana aprendida somente nos dados de treino com `SimpleImputer`.
- Variáveis numéricas são padronizadas com `StandardScaler`.
- Variáveis categóricas são codificadas com `OneHotEncoder`.
- Todas as etapas são reunidas em uma `Pipeline`, que recebe os dados crus e aplica o mesmo tratamento no treino e no teste.

## Modelos avaliados

- **Logistic Regression**
- **Decision Tree sem limite de profundidade**
- **Decision Tree controlada**, com `max_depth=4`

Na validação cruzada, a árvore sem controle apresentou forte overfitting: accuracy média de aproximadamente 99,84% no treino e 72,01% na validação. A árvore controlada reduziu bastante essa diferença, mas a Logistic Regression apresentou resultados mais consistentes e foi mantida como modelo final.

## Modelo escolhido

O modelo final é a **Logistic Regression**. Além do desempenho observado, ela é uma opção simples e adequada ao objetivo deste primeiro projeto de classificação.

## Resultados finais

Os resultados abaixo foram calculados no conjunto de teste, que não foi usado para treinar o modelo.

| Métrica | Resultado |
|---|---:|
| Accuracy | 80,55% |
| Recall | 55,88% |
| Precision | 65,72% |
| F1 | 60,40% |

Matriz de confusão:

|  | Previsto: não churn | Previsto: churn |
|---|---:|---:|
| **Real: não churn** | TN = 926 | FP = 109 |
| **Real: churn** | FN = 165 | TP = 209 |

## Interpretação

- **Recall de aproximadamente 56%:** entre os clientes que realmente cancelaram, o modelo identificou cerca de 56%.
- **Precision de aproximadamente 66%:** entre os clientes classificados pelo modelo como churn, cerca de 66% realmente cancelaram.
- **F1 de aproximadamente 60%:** resume o equilíbrio entre Precision e Recall, sem definir sozinho se o modelo é adequado para um negócio específico.
- A principal limitação observada são os **165 falsos negativos**: clientes que cancelaram, mas foram classificados como não-churn.

## Limitações

- O Recall ainda é limitado e existe uma quantidade relevante de falsos negativos.
- O modelo não deve ser apresentado como pronto para produção.
- A escolha da métrica mais importante depende dos custos e objetivos reais do negócio.
- O estudo usa uma única divisão treino/teste para a avaliação final e um dataset de exemplo.

## Próximos passos

Possíveis estudos futuros incluem avaliar outros thresholds, testar novos modelos, criar novas variáveis e analisar o custo de falsos positivos e falsos negativos. Essas melhorias são apenas possibilidades e **não foram implementadas neste projeto**.

## Como executar

Recomenda-se Python 3.12.

```bash
python -m venv .venv
```

Ative o ambiente virtual:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

Instale as dependências e abra o notebook:

```bash
python -m pip install -r requirements.txt
python -m notebook notebooks/customer_churn_analysis.ipynb
```

No Jupyter, execute todas as células em ordem. O notebook lê o CSV por caminho relativo e não depende de uma pasta específica do computador.
