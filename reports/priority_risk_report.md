# Relatório de Priorização de Risco de Churn (Fora da Amostra de Treino)

Relatório gerado automaticamente por `src/analytics/risk_ranking.py` usando a pipeline final (`Logistic Regression`) aplicada exclusivamente aos **1.409 clientes do conjunto de teste** (`test_size=0.20`, `random_state=42`, `stratify=y`), evitando o viés otimista de pontuar os mesmos clientes usados no treinamento.

> **Avisos de interpretação obrigatórios:**
> 1. **Probabilidade estimada não é certeza de cancelamento:** Uma probabilidade estimada de `0,80` significa que, segundo o modelo, aquele perfil se assemelha a casos históricos de alto risco — não garante que o cliente cancelará.
> 2. **Receita mensal associada não é perda financeira comprovada:** A soma de `MonthlyCharges` dos clientes sinalizados mede apenas a receita mensal atualmente vinculada a esse grupo. Ela **não** representa perda financeira esperada garantida nem economia comprovada de campanhas de retenção.
> 3. **Faixas operacionais vs. classificação binária:** As faixas `Alto` (`>= 0,60`), `Moderado` (`0,35` a `0,5999`) e `Baixo` (`< 0,35`) servem apenas para organizar uma fila de contato. A previsão binária (`Yes`/`No`) do modelo continua inalterada no threshold padrão `0,50`.

## 1. Resumo dos Clientes Sinalizados (`previsao = Yes`)

| Indicador (Conjunto de Teste — 1.409 clientes) | Valor |
| --- | ---: |
| Total de clientes avaliados fora do treino | 1,409 |
| Clientes sinalizados como `Yes` (threshold padrão 0,50) | 318 (22.57%) |
| Receita mensal total da amostra avaliada | US$ 90,301.20 |
| Receita mensal associada aos clientes sinalizados (`Yes`) | US$ 25,211.65 |
| Cobrança mensal média dos clientes sinalizados (`Yes`) | US$ 79.28 |

## 2. Distribuição por Faixa Operacional de Risco

| Faixa Operacional | Critério (`probabilidade_churn`) | Clientes | % da Amostra | Prob. Estimada Média | Churn Real Histórico | Taxa Real no Grupo | Receita Mensal Associada (US$) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Alto** | `>= 0.60` | 209 | 14.83% | 69.40% | 150 | 71.77% | 17,051.85 |
| **Moderado** | `0.35` a `0.5999` | 276 | 19.59% | 47.72% | 114 | 41.30% | 20,537.75 |
| **Baixo** | `< 0.35` | 924 | 65.58% | 10.89% | 110 | 11.90% | 52,711.60 |

## 3. Top 20 Clientes na Fila de Priorização (Amostra de Teste)

| Posição | `customerID` | Prob. Estimada | Previsão | Faixa | Contrato | Internet | Tenure | Cobrança Mensal (US$) | Churn Real |
| ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | `5178-LMXOP` | 85.49% | `Yes` | Alto | Month-to-month | Fiber optic | 1 | 95.10 | `Yes` |
| 2 | `0295-PPHDO` | 83.20% | `Yes` | Alto | Month-to-month | Fiber optic | 1 | 95.45 | `Yes` |
| 3 | `6861-XWTWQ` | 82.86% | `Yes` | Alto | Month-to-month | Fiber optic | 7 | 99.25 | `Yes` |
| 4 | `8884-ADFVN` | 82.22% | `Yes` | Alto | Month-to-month | Fiber optic | 7 | 101.95 | `Yes` |
| 5 | `1069-XAIEM` | 81.88% | `Yes` | Alto | Month-to-month | Fiber optic | 1 | 85.05 | `Yes` |
| 6 | `6023-YEBUP` | 81.78% | `Yes` | Alto | Month-to-month | Fiber optic | 3 | 100.95 | `Yes` |
| 7 | `2545-EBUPK` | 81.75% | `Yes` | Alto | Month-to-month | Fiber optic | 2 | 84.05 | `No` |
| 8 | `9057-SIHCH` | 81.74% | `Yes` | Alto | Month-to-month | Fiber optic | 3 | 96.60 | `Yes` |
| 9 | `1400-MMYXY` | 81.67% | `Yes` | Alto | Month-to-month | Fiber optic | 3 | 105.90 | `Yes` |
| 10 | `6630-UJZMY` | 81.18% | `Yes` | Alto | Month-to-month | Fiber optic | 4 | 83.25 | `No` |
| 11 | `8161-QYMTT` | 81.16% | `Yes` | Alto | Month-to-month | Fiber optic | 7 | 94.10 | `No` |
| 12 | `4750-ZRXIU` | 80.65% | `Yes` | Alto | Month-to-month | Fiber optic | 4 | 84.60 | `Yes` |
| 13 | `2865-TCHJW` | 80.06% | `Yes` | Alto | Month-to-month | Fiber optic | 4 | 89.20 | `Yes` |
| 14 | `3320-VEOYC` | 79.48% | `Yes` | Alto | Month-to-month | Fiber optic | 14 | 95.60 | `No` |
| 15 | `3722-WPXTK` | 79.47% | `Yes` | Alto | Month-to-month | Fiber optic | 1 | 88.35 | `Yes` |
| 16 | `0970-ETWGE` | 79.09% | `Yes` | Alto | Month-to-month | Fiber optic | 1 | 89.55 | `Yes` |
| 17 | `8593-WHYHV` | 78.90% | `Yes` | Alto | Month-to-month | Fiber optic | 3 | 85.30 | `Yes` |
| 18 | `1143-NMNQJ` | 78.89% | `Yes` | Alto | Month-to-month | Fiber optic | 2 | 85.70 | `Yes` |
| 19 | `4927-WWOOZ` | 78.66% | `Yes` | Alto | Month-to-month | Fiber optic | 2 | 91.45 | `No` |
| 20 | `8245-UMPYT` | 78.33% | `Yes` | Alto | Month-to-month | Fiber optic | 16 | 96.40 | `Yes` |

## 4. Como Usar a Fila na Prática (e Limitações)

- Em vez de agir com a mesma prioridade sobre todos os clientes, uma equipe operacional pode ordenar o atendimento pelo score estimado de risco, combinando o nível de risco com a capacidade real de contato.
- A existência de falsos positivos (clientes sinalizados que não cancelaram) e falsos negativos (165 clientes que cancelaram e ficaram abaixo do corte `0,50`) reforça que o score é um apoio à priorização, não uma decisão automática infalível.
- A tabela dos clientes ranqueados pode ser gerada localmente ao executar a análise de risco; registros individuais não são distribuídos nesta versão.
