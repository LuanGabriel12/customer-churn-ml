"""Analises comerciais de churn usando apenas Python basico e Pandas."""

from pathlib import Path

import pandas as pd

from src.config import (
    ARQUIVO_CHURN_SEGMENTOS,
    ARQUIVO_DADOS_BRUTOS,
    ARQUIVO_RELATORIO_COMERCIAL,
    CLASSE_POSITIVA,
    COLUNA_ALVO,
)
from src.data.loader import carregar_dataset


def classificar_faixa_tenure(meses: int) -> str:
    """Agrupa o tempo de permanencia (em meses) em faixas interpretaveis."""
    if meses <= 12:
        return "01. 0 a 12 meses"
    if meses <= 24:
        return "02. 13 a 24 meses"
    if meses <= 48:
        return "03. 25 a 48 meses"
    return "04. 49 a 72 meses"


def classificar_faixa_cobranca(valor_mensal: float) -> str:
    """Agrupa a cobranca mensal (MonthlyCharges) em faixas de preco."""
    if valor_mensal <= 35.0:
        return "01. Ate US$ 35,00"
    if valor_mensal <= 70.0:
        return "02. US$ 35,01 a US$ 70,00"
    if valor_mensal <= 90.0:
        return "03. US$ 70,01 a US$ 90,00"
    return "04. Acima de US$ 90,00"


def preparar_base_analitica(dados: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas auxiliares de faixas e rotulo legivel para analise comercial."""
    df = dados.copy()
    df["faixa_tenure"] = [
        classificar_faixa_tenure(int(meses)) for meses in df["tenure"].tolist()
    ]
    df["faixa_cobranca_mensal"] = [
        classificar_faixa_cobranca(float(valor))
        for valor in df["MonthlyCharges"].tolist()
    ]
    df["perfil_senior"] = [
        "Idoso (SeniorCitizen=1)" if int(valor) == 1 else "Nao idoso (SeniorCitizen=0)"
        for valor in df["SeniorCitizen"].tolist()
    ]
    df["contrato_x_internet"] = (
        df["Contract"].astype(str) + " + " + df["InternetService"].astype(str)
    )
    return df


def calcular_churn_por_categoria(
    dados: pd.DataFrame,
    coluna: str,
    nome_dimensao: str,
    ordenar_por_taxa: bool = True,
) -> pd.DataFrame:
    """Calcula volume de clientes, taxa de churn e cobranca mensal por categoria."""
    categorias = sorted(dados[coluna].dropna().unique().tolist())
    linhas = []

    for categoria in categorias:
        grupo = dados[dados[coluna] == categoria]
        total_clientes = int(len(grupo))
        total_churn = int((grupo[COLUNA_ALVO] == CLASSE_POSITIVA).sum())
        taxa_churn = (total_churn / total_clientes) * 100 if total_clientes > 0 else 0.0
        receita_mensal_total = float(grupo["MonthlyCharges"].sum())
        receita_mensal_media = (
            float(grupo["MonthlyCharges"].mean()) if total_clientes > 0 else 0.0
        )
        receita_mensal_churn = float(
            grupo[grupo[COLUNA_ALVO] == CLASSE_POSITIVA]["MonthlyCharges"].sum()
        )
        tenure_medio = float(grupo["tenure"].mean()) if total_clientes > 0 else 0.0

        linhas.append(
            {
                "dimensao": nome_dimensao,
                "segmento": str(categoria),
                "clientes": total_clientes,
                "clientes_churn": total_churn,
                "taxa_churn_pct": round(taxa_churn, 2),
                "tenure_medio_meses": round(tenure_medio, 2),
                "cobranca_mensal_media": round(receita_mensal_media, 2),
                "receita_mensal_total": round(receita_mensal_total, 2),
                "receita_mensal_clientes_churn": round(receita_mensal_churn, 2),
            }
        )

    tabela = pd.DataFrame(linhas)
    if ordenar_por_taxa and not tabela.empty:
        tabela = tabela.sort_values(
            by=["taxa_churn_pct", "clientes"], ascending=[False, False]
        ).reset_index(drop=True)
    return tabela


def analisar_segmentos_clientes(dados: pd.DataFrame) -> dict:
    """Executa todas as analises comerciais por contrato, internet, tenure, cobranca e segmentos."""
    base = preparar_base_analitica(dados)

    total_clientes = int(len(base))
    total_churn = int((base[COLUNA_ALVO] == CLASSE_POSITIVA).sum())
    taxa_geral = round((total_churn / total_clientes) * 100, 2) if total_clientes > 0 else 0.0
    receita_mensal_base = round(float(base["MonthlyCharges"].sum()), 2)
    receita_mensal_churn_hist = round(
        float(base[base[COLUNA_ALVO] == CLASSE_POSITIVA]["MonthlyCharges"].sum()), 2
    )

    tabelas = {
        "contrato": calcular_churn_por_categoria(
            base, "Contract", "Tipo de Contrato", ordenar_por_taxa=True
        ),
        "internet": calcular_churn_por_categoria(
            base, "InternetService", "Servico de Internet", ordenar_por_taxa=True
        ),
        "tenure": calcular_churn_por_categoria(
            base, "faixa_tenure", "Faixa de Permanencia", ordenar_por_taxa=False
        ),
        "cobranca_mensal": calcular_churn_por_categoria(
            base,
            "faixa_cobranca_mensal",
            "Faixa de Cobranca Mensal",
            ordenar_por_taxa=False,
        ),
        "pagamento": calcular_churn_por_categoria(
            base, "PaymentMethod", "Forma de Pagamento", ordenar_por_taxa=True
        ),
        "senioridade": calcular_churn_por_categoria(
            base, "perfil_senior", "Faixa Etaria (SeniorCitizen)", ordenar_por_taxa=True
        ),
        "fatura_digital": calcular_churn_por_categoria(
            base, "PaperlessBilling", "Fatura Digital", ordenar_por_taxa=True
        ),
        "contrato_x_internet": calcular_churn_por_categoria(
            base,
            "contrato_x_internet",
            "Contrato x Internet",
            ordenar_por_taxa=True,
        ),
    }

    return {
        "resumo_geral": {
            "total_clientes": total_clientes,
            "clientes_churn": total_churn,
            "taxa_churn_geral_pct": taxa_geral,
            "receita_mensal_total": receita_mensal_base,
            "receita_mensal_associada_churn_historico": receita_mensal_churn_hist,
            "cobranca_mensal_media_geral": round(
                float(base["MonthlyCharges"].mean()), 2
            ),
            "cobranca_mensal_media_churn": round(
                float(
                    base[base[COLUNA_ALVO] == CLASSE_POSITIVA]["MonthlyCharges"].mean()
                ),
                2,
            ),
            "cobranca_mensal_media_nao_churn": round(
                float(base[base[COLUNA_ALVO] != CLASSE_POSITIVA]["MonthlyCharges"].mean()),
                2,
            ),
            "tenure_medio_churn": round(
                float(base[base[COLUNA_ALVO] == CLASSE_POSITIVA]["tenure"].mean()), 2
            ),
            "tenure_medio_nao_churn": round(
                float(base[base[COLUNA_ALVO] != CLASSE_POSITIVA]["tenure"].mean()), 2
            ),
        },
        "tabelas": tabelas,
    }


def _tabela_para_markdown(df: pd.DataFrame) -> list[str]:
    """Formata um DataFrame de segmento como tabela Markdown limpa."""
    linhas = [
        "| Segmento | Clientes | Churn | Taxa de Churn | Tenure Médio (meses) | Cobrança Média (US$) | Receita Mensal Total (US$) | Receita Mensal em Churn (US$) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for registro in df.to_dict(orient="records"):
        linhas.append(
            f"| {registro['segmento']} | {registro['clientes']:,} | {registro['clientes_churn']:,} | "
            f"{registro['taxa_churn_pct']:.2f}% | {registro['tenure_medio_meses']:.2f} | "
            f"{registro['cobranca_mensal_media']:,.2f} | {registro['receita_mensal_total']:,.2f} | "
            f"{registro['receita_mensal_clientes_churn']:,.2f} |"
        )
    return linhas


def gerar_relatorio_comercial_markdown(analise: dict) -> str:
    """Constroi o relatorio executivo comercial em Markdown com rigor estatistico."""
    resumo = analise["resumo_geral"]
    tabelas = analise["tabelas"]

    linhas = [
        "# Relatório de Inteligência Comercial — Análise de Churn",
        "",
        "Relatório gerado automaticamente por `src/analytics/customer_analysis.py` sobre a base completa **IBM Telco Customer Churn** (`7.043` clientes).",
        "",
        "> **Aviso metodológico obrigatório:** As comparações abaixo descrevem **associações históricas observadas** no dataset. Elas **não comprovam relações de causa e efeito** (por exemplo, ter fibra óptica ou contrato mensal está associado a maior taxa de churn na base, mas não prova isoladamente o motivo real do cancelamento). Da mesma forma, valores de `MonthlyCharges` representam **receita mensal associada aos grupos observados**, e **não** perda financeira líquida nem retorno comprovado de ações de retenção.",
        "",
        "## 1. Panorama Geral da Base",
        "",
        "| Indicador | Valor Observado |",
        "| --- | ---: |",
        f"| Total de clientes | {resumo['total_clientes']:,} |",
        f"| Clientes com churn histórico (`Churn = Yes`) | {resumo['clientes_churn']:,} |",
        f"| Taxa geral de churn | {resumo['taxa_churn_geral_pct']:.2f}% |",
        f"| Receita mensal total da base (`MonthlyCharges`) | US$ {resumo['receita_mensal_total']:,.2f} |",
        f"| Receita mensal associada aos clientes que cancelaram | US$ {resumo['receita_mensal_associada_churn_historico']:,.2f} |",
        f"| Cobrança mensal média (churn vs. não-churn) | US$ {resumo['cobranca_mensal_media_churn']:.2f} vs. US$ {resumo['cobranca_mensal_media_nao_churn']:.2f} |",
        f"| Tempo médio de permanência (churn vs. não-churn) | {resumo['tenure_medio_churn']:.2f} vs. {resumo['tenure_medio_nao_churn']:.2f} meses |",
        "",
        "## 2. Taxa de Churn por Tipo de Contrato",
        "",
    ]
    linhas.extend(_tabela_para_markdown(tabelas["contrato"]))

    linhas.extend(
        [
            "",
            "## 3. Taxa de Churn por Serviço de Internet",
            "",
        ]
    )
    linhas.extend(_tabela_para_markdown(tabelas["internet"]))

    linhas.extend(
        [
            "",
            "## 4. Análise por Tempo de Permanência (`tenure`)",
            "",
        ]
    )
    linhas.extend(_tabela_para_markdown(tabelas["tenure"]))

    linhas.extend(
        [
            "",
            "## 5. Distribuição por Cobrança Mensal (`MonthlyCharges`)",
            "",
        ]
    )
    linhas.extend(_tabela_para_markdown(tabelas["cobranca_mensal"]))

    linhas.extend(
        [
            "",
            "## 6. Comparação por Forma de Pagamento",
            "",
        ]
    )
    linhas.extend(_tabela_para_markdown(tabelas["pagamento"]))

    linhas.extend(
        [
            "",
            "## 7. Comparação por Perfil Etário (`SeniorCitizen`) e Fatura Digital",
            "",
        ]
    )
    linhas.extend(_tabela_para_markdown(tabelas["senioridade"]))
    linhas.append("")
    linhas.extend(_tabela_para_markdown(tabelas["fatura_digital"]))

    linhas.extend(
        [
            "",
            "## 8. Segmentos Combinados: Contrato × Serviço de Internet",
            "",
        ]
    )
    linhas.extend(_tabela_para_markdown(tabelas["contrato_x_internet"]))

    linhas.extend(
        [
            "",
            "## 9. Síntese Executiva (Associações Observadas)",
            "",
            "- **Contrato:** Clientes `Month-to-month` concentram a maior taxa de churn associada da base, muito acima dos contratos `One year` e `Two year`.",
            "- **Serviço de Internet:** Clientes com `Fiber optic` apresentam taxa de churn associada significativamente superior a `DSL` e a clientes sem serviço de internet.",
            "- **Tempo de permanência:** A maior taxa de cancelamento ocorre nos primeiros **0 a 12 meses**, caindo progressivamente em faixas mais longas de relacionamento.",
            "- **Cobrança mensal:** Faixas de cobrança acima de **US$ 70,00/mês** apresentam maior taxa de churn associada e maior ticket médio; porém, parte desse padrão coincide com planos de `Fiber optic` e contratos mensais (associação, não causalidade isolada).",
            "- **Cruzamento de segmentos:** A combinação `Month-to-month + Fiber optic` reúne o maior risco histórico observado, enquanto `Two year + No` apresenta a menor taxa de churn.",
            "",
        ]
    )

    return "\n".join(linhas)


def exportar_relatorio_comercial(
    caminho_dados: Path = ARQUIVO_DADOS_BRUTOS,
    caminho_markdown: Path = ARQUIVO_RELATORIO_COMERCIAL,
    caminho_csv: Path = ARQUIVO_CHURN_SEGMENTOS,
) -> dict:
    """Executa a analise comercial completa e exporta CSV consolidado e Markdown."""
    dados = carregar_dataset(caminho_dados)
    analise = analisar_segmentos_clientes(dados)

    tabelas_concatenadas = pd.concat(
        list(analise["tabelas"].values()), ignore_index=True
    )
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    tabelas_concatenadas.to_csv(caminho_csv, index=False)

    relatorio_md = gerar_relatorio_comercial_markdown(analise)
    caminho_markdown.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_markdown, "w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio_md)

    return analise


if __name__ == "__main__":
    resultado = exportar_relatorio_comercial()
    resumo = resultado["resumo_geral"]
    print(
        f"Analise comercial concluida: {resumo['total_clientes']} clientes, "
        f"taxa geral de churn = {resumo['taxa_churn_geral_pct']:.2f}%."
    )
    print(f"Relatorio salvo em: {ARQUIVO_RELATORIO_COMERCIAL}")
    print(f"CSV consolidado salvo em: {ARQUIVO_CHURN_SEGMENTOS}")
