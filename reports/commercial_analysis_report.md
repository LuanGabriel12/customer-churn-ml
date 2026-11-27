# Relatório de Inteligência Comercial — Análise de Churn

Relatório gerado automaticamente por `src/analytics/customer_analysis.py` sobre a base completa **IBM Telco Customer Churn** (`7.043` clientes).

> **Aviso metodológico obrigatório:** As comparações abaixo descrevem **associações históricas observadas** no dataset. Elas **não comprovam relações de causa e efeito** (por exemplo, ter fibra óptica ou contrato mensal está associado a maior taxa de churn na base, mas não prova isoladamente o motivo real do cancelamento). Da mesma forma, valores de `MonthlyCharges` representam **receita mensal associada aos grupos observados**, e **não** perda financeira líquida nem retorno comprovado de ações de retenção.

## 1. Panorama Geral da Base

| Indicador | Valor Observado |
| --- | ---: |
| Total de clientes | 7,043 |
| Clientes com churn histórico (`Churn = Yes`) | 1,869 |
| Taxa geral de churn | 26.54% |
| Receita mensal total da base (`MonthlyCharges`) | US$ 456,116.60 |
| Receita mensal associada aos clientes que cancelaram | US$ 139,130.85 |
| Cobrança mensal média (churn vs. não-churn) | US$ 74.44 vs. US$ 61.27 |
| Tempo médio de permanência (churn vs. não-churn) | 17.98 vs. 37.57 meses |

## 2. Taxa de Churn por Tipo de Contrato

| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Month-to-month | 3,875 | 1,655 | 42.71% | 18.04 | 66.40 | 257,294.15 | 120,847.10 |
| One year | 1,473 | 166 | 11.27% | 42.04 | 65.05 | 95,816.60 | 14,118.45 |
| Two year | 1,695 | 48 | 2.83% | 56.74 | 60.77 | 103,005.85 | 4,165.30 |

## 3. Taxa de Churn por Serviço de Internet

| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fiber optic | 3,096 | 1,297 | 41.89% | 32.92 | 91.50 | 283,284.40 | 114,300.05 |
| DSL | 2,421 | 459 | 18.96% | 32.82 | 58.10 | 140,665.35 | 22,529.20 |
| No | 1,526 | 113 | 7.40% | 30.55 | 21.08 | 32,166.85 | 2,301.60 |

## 4. Análise por Tempo de Permanência (`tenure`)

| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01. 0 a 12 meses | 2,186 | 1,037 | 47.44% | 4.73 | 56.10 | 122,629.75 | 68,954.25 |
| 02. 13 a 24 meses | 1,024 | 294 | 28.71% | 18.35 | 61.36 | 62,829.85 | 23,081.65 |
| 03. 25 a 48 meses | 1,594 | 325 | 20.39% | 36.19 | 65.93 | 105,093.30 | 27,462.50 |
| 04. 49 a 72 meses | 2,239 | 213 | 9.51% | 63.05 | 73.95 | 165,563.70 | 19,632.45 |

## 5. Distribuição por Cobrança Mensal (`MonthlyCharges`)

| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01. Ate US$ 35,00 | 1,735 | 189 | 10.89% | 28.67 | 22.03 | 38,217.15 | 4,458.05 |
| 02. US$ 35,01 a US$ 70,00 | 1,725 | 413 | 23.94% | 26.87 | 54.84 | 94,597.15 | 22,181.80 |
| 03. US$ 70,01 a US$ 90,00 | 1,844 | 697 | 37.80% | 29.23 | 80.06 | 147,630.80 | 55,706.90 |
| 04. Acima de US$ 90,00 | 1,739 | 570 | 32.78% | 44.84 | 101.02 | 175,671.50 | 56,784.10 |

## 6. Comparação por Forma de Pagamento

| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Electronic check | 2,365 | 1,071 | 45.29% | 25.17 | 76.26 | 180,345.00 | 84,288.75 |
| Mailed check | 1,612 | 308 | 19.11% | 21.83 | 43.92 | 70,794.30 | 16,803.60 |
| Bank transfer (automatic) | 1,544 | 258 | 16.71% | 43.66 | 67.19 | 103,745.45 | 20,091.90 |
| Credit card (automatic) | 1,522 | 232 | 15.24% | 43.27 | 66.51 | 101,231.85 | 17,946.60 |

## 7. Comparação por Perfil Etário (`SeniorCitizen`) e Fatura Digital

| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Idoso (SeniorCitizen=1) | 1,142 | 476 | 41.68% | 33.30 | 79.82 | 91,154.85 | 38,419.60 |
| Nao idoso (SeniorCitizen=0) | 5,901 | 1,393 | 23.61% | 32.19 | 61.85 | 364,961.75 | 100,711.25 |

| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Yes | 4,171 | 1,400 | 33.57% | 32.50 | 73.55 | 306,792.80 | 109,510.00 |
| No | 2,872 | 469 | 16.33% | 32.19 | 51.99 | 149,323.80 | 29,620.85 |

## 8. Segmentos Combinados: Contrato × Serviço de Internet

| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Month-to-month + Fiber optic | 2,128 | 1,162 | 54.61% | 21.55 | 87.02 | 185,181.10 | 100,482.00 |
| Month-to-month + DSL | 1,223 | 394 | 32.22% | 15.73 | 50.22 | 61,418.45 | 18,367.25 |
| One year + Fiber optic | 539 | 104 | 19.29% | 52.10 | 98.78 | 53,242.15 | 10,571.95 |
| Month-to-month + No | 524 | 99 | 18.89% | 9.13 | 20.41 | 10,694.60 | 1,997.85 |
| One year + DSL | 570 | 53 | 9.30% | 40.49 | 61.40 | 34,996.15 | 3,356.25 |
| Two year + Fiber optic | 429 | 31 | 7.23% | 65.18 | 104.57 | 44,861.15 | 3,246.10 |
| One year + No | 364 | 9 | 2.47% | 29.60 | 20.82 | 7,578.30 | 190.25 |
| Two year + DSL | 628 | 12 | 1.91% | 59.14 | 70.46 | 44,250.75 | 805.70 |
| Two year + No | 638 | 5 | 0.78% | 48.68 | 21.78 | 13,893.95 | 113.50 |

## 9. Síntese Executiva (Associações Observadas)

- **Contrato:** Clientes `Month-to-month` concentram a maior taxa de churn associada da base, muito acima dos contratos `One year` e `Two year`.
- **Serviço de Internet:** Clientes com `Fiber optic` apresentam taxa de churn associada significativamente superior a `DSL` e a clientes sem serviço de internet.
- **Tempo de permanência:** A maior taxa de cancelamento ocorre nos primeiros **0 a 12 meses**, caindo progressivamente em faixas mais longas de relacionamento.
- **Cobrança mensal:** Faixas de cobrança acima de **US$ 70,00/mês** apresentam maior taxa de churn associada e maior ticket médio; porém, parte desse padrão coincide com planos de `Fiber optic` e contratos mensais (associação, não causalidade isolada).
- **Cruzamento de segmentos:** A combinação `Month-to-month + Fiber optic` reúne o maior risco histórico observado, enquanto `Two year + No` apresenta a menor taxa de churn.
