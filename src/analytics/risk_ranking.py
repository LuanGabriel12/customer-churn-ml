"""Fila de priorizacao de risco de churn fora da amostra de treino (conjunto de teste)."""

from pathlib import Path

import pandas as pd

from src.config import (
    ARQUIVO_DADOS_BRUTOS,
    ARQUIVO_FILA_PRIORITARIA,
    ARQUIVO_MODELO,
    ARQUIVO_RELATORIO_RISCO,
    CLASSE_POSITIVA,
    COLUNA_ALVO,
    COLUNA_ID,
    LIMITE_RISCO_ALTO,
    LIMITE_RISCO_MODERADO,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data.loader import (
    carregar_dataset,
    dividir_treino_e_teste,
    separar_features_e_alvo,
)
from src.modeling.predict import (
    carregar_modelo,
    classificar_faixa_risco,
    prever_lote,
)


def gerar_fila_priorizacao(
    modelo,
    dados_com_id: pd.DataFrame,
    X_para_pontuar: pd.DataFrame,
    y_real: pd.Series | None = None,
) -> pd.DataFrame:
    """Pontua clientes, ordena por probabilidade estimada de churn e atribui faixa operacional."""
    registros = X_para_pontuar.to_dict(orient="records")
    previsoes = prever_lote(modelo, registros)

    indices = list(X_para_pontuar.index)
    linhas = []
    for posicao, indice_original in enumerate(indices):
        pred = previsoes[posicao]
        prob_churn = pred["probabilidade_churn"]
        linha_dados = dados_com_id.loc[indice_original]

        item = {
            "customerID": str(linha_dados[COLUNA_ID]),
            "Contract": str(linha_dados["Contract"]),
            "InternetService": str(linha_dados["InternetService"]),
            "PaymentMethod": str(linha_dados["PaymentMethod"]),
            "tenure": int(linha_dados["tenure"]),
            "MonthlyCharges": round(float(linha_dados["MonthlyCharges"]), 2),
            "previsao": pred["previsao"],
            "probabilidade_churn": round(prob_churn, 6),
            "probabilidade_nao_churn": round(pred["probabilidade_nao_churn"], 6),
            "faixa_risco": classificar_faixa_risco(prob_churn),
        }
        if y_real is not None:
            item["churn_real_historico"] = str(y_real.loc[indice_original])
        linhas.append(item)

    fila = pd.DataFrame(linhas)
    fila = fila.sort_values(
        by=["probabilidade_churn", "MonthlyCharges"], ascending=[False, False]
    ).reset_index(drop=True)
    fila.insert(0, "posicao_fila", list(range(1, len(fila) + 1)))
    return fila


def resumir_fila_risco(fila: pd.DataFrame) -> dict:
    """Resume volume, receita mensal associada e diagnostico por faixa operacional."""
    total_avaliados = int(len(fila))
    sinalizados = fila[fila["previsao"] == CLASSE_POSITIVA]
    total_sinalizados = int(len(sinalizados))

    receita_total_avaliados = round(float(fila["MonthlyCharges"].sum()), 2)
    receita_sinalizados = round(float(sinalizados["MonthlyCharges"].sum()), 2)
    ticket_medio_sinalizados = (
        round(float(sinalizados["MonthlyCharges"].mean()), 2)
        if total_sinalizados > 0
        else 0.0
    )

    faixas = []
    for nome_faixa in ["Alto", "Moderado", "Baixo"]:
        grupo = fila[fila["faixa_risco"] == nome_faixa]
        qtd = int(len(grupo))
        pct_base = round((qtd / total_avaliados) * 100, 2) if total_avaliados > 0 else 0.0
        prob_media = (
            round(float(grupo["probabilidade_churn"].mean()) * 100, 2)
            if qtd > 0
            else 0.0
        )
        receita_faixa = round(float(grupo["MonthlyCharges"].sum()), 2)

        item_faixa = {
            "faixa_risco": nome_faixa,
            "clientes": qtd,
            "participacao_pct": pct_base,
            "probabilidade_media_pct": prob_media,
            "receita_mensal_associada": receita_faixa,
        }
        if "churn_real_historico" in grupo.columns:
            churns_reais = int((grupo["churn_real_historico"] == CLASSE_POSITIVA).sum())
            item_faixa["churn_real_historico"] = churns_reais
            item_faixa["taxa_churn_real_pct"] = (
                round((churns_reais / qtd) * 100, 2) if qtd > 0 else 0.0
            )
        faixas.append(item_faixa)

    return {
        "total_avaliados": total_avaliados,
        "total_sinalizados_yes": total_sinalizados,
        "percentual_sinalizados": round((total_sinalizados / total_avaliados) * 100, 2)
        if total_avaliados > 0
        else 0.0,
        "receita_mensal_total_avaliados": receita_total_avaliados,
        "receita_mensal_associada_sinalizados": receita_sinalizados,
        "cobranca_mensal_media_sinalizados": ticket_medio_sinalizados,
        "resumo_por_faixa": faixas,
    }


def gerar_relatorio_risco_markdown(fila: pd.DataFrame, resumo: dict) -> str:
    """Gera o relatorio executivo da fila de priorizacao em Markdown."""
    linhas = [
        "# Relatório de Priorização de Risco de Churn (Fora da Amostra de Treino)",
        "",
        "Relatório gerado automaticamente por `src/analytics/risk_ranking.py` usando a pipeline final (`Logistic Regression`) aplicada exclusivamente aos **1.409 clientes do conjunto de teste** (`test_size=0.20`, `random_state=42`, `stratify=y`), evitando o viés otimista de pontuar os mesmos clientes usados no treinamento.",
        "",
        "> **Avisos de interpretação obrigatórios:**",
        "> 1. **Probabilidade estimada não é certeza de cancelamento:** Uma probabilidade estimada de `0,80` significa que, segundo o modelo, aquele perfil se assemelha a casos históricos de alto risco — não garante que o cliente cancelará.",
        "> 2. **Receita mensal associada não é perda financeira comprovada:** A soma de `MonthlyCharges` dos clientes sinalizados mede apenas a receita mensal atualmente vinculada a esse grupo. Ela **não** representa perda financeira esperada garantida nem economia comprovada de campanhas de retenção.",
        "> 3. **Faixas operacionais vs. classificação binária:** As faixas `Alto` (`>= 0,60`), `Moderado` (`0,35` a `0,5999`) e `Baixo` (`< 0,35`) servem apenas para organizar uma fila de contato. A previsão binária (`Yes`/`No`) do modelo continua inalterada no threshold padrão `0,50`.",
        "",
        "## 1. Resumo dos Clientes Sinalizados (`previsao = Yes`)",
        "",
        "| Indicador (Conjunto de Teste — 1.409 clientes) | Valor |",
        "| --- | ---: |",
        f"| Total de clientes avaliados fora do treino | {resumo['total_avaliados']:,} |",
        f"| Clientes sinalizados como `Yes` (threshold padrão 0,50) | {resumo['total_sinalizados_yes']:,} ({resumo['percentual_sinalizados']:.2f}%) |",
        f"| Receita mensal total da amostra avaliada | US$ {resumo['receita_mensal_total_avaliados']:,.2f} |",
        f"| Receita mensal associada aos clientes sinalizados (`Yes`) | US$ {resumo['receita_mensal_associada_sinalizados']:,.2f} |",
        f"| Cobrança mensal média dos clientes sinalizados (`Yes`) | US$ {resumo['cobranca_mensal_media_sinalizados']:,.2f} |",
        "",
        "## 2. Distribuição por Faixa Operacional de Risco",
        "",
        "| Faixa Operacional | Critério (`probabilidade_churn`) | Clientes | % da Amostra | Prob. Estimada Média | Churn Real Histórico | Taxa Real no Grupo | Receita Mensal Associada (US$) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    criterios = {
        "Alto": f"`>= {LIMITE_RISCO_ALTO:.2f}`",
        "Moderado": f"`{LIMITE_RISCO_MODERADO:.2f}` a `{LIMITE_RISCO_ALTO - 0.0001:.4f}`",
        "Baixo": f"`< {LIMITE_RISCO_MODERADO:.2f}`",
    }
    for faixa in resumo["resumo_por_faixa"]:
        linhas.append(
            f"| **{faixa['faixa_risco']}** | {criterios[faixa['faixa_risco']]} | "
            f"{faixa['clientes']:,} | {faixa['participacao_pct']:.2f}% | "
            f"{faixa['probabilidade_media_pct']:.2f}% | "
            f"{faixa.get('churn_real_historico', 0):,} | "
            f"{faixa.get('taxa_churn_real_pct', 0.0):.2f}% | "
            f"{faixa['receita_mensal_associada']:,.2f} |"
        )

    linhas.extend(
        [
            "",
            "## 3. Top 20 Clientes na Fila de Priorização (Amostra de Teste)",
            "",
            "| Posição | `customerID` | Prob. Estimada | Previsão | Faixa | Contrato | Internet | Tenure | Cobrança Mensal (US$) | Churn Real |",
            "| ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )

    top_20 = fila.head(20)
    for registro in top_20.to_dict(orient="records"):
        linhas.append(
            f"| {registro['posicao_fila']} | `{registro['customerID']}` | "
            f"{registro['probabilidade_churn'] * 100:.2f}% | `{registro['previsao']}` | "
            f"{registro['faixa_risco']} | {registro['Contract']} | {registro['InternetService']} | "
            f"{registro['tenure']} | {registro['MonthlyCharges']:.2f} | "
            f"`{registro.get('churn_real_historico', '-')}` |"
        )

    linhas.extend(
        [
            "",
            "## 4. Como Usar a Fila na Prática (e Limitações)",
            "",
            "- Em vez de agir com a mesma prioridade sobre todos os clientes, uma equipe operacional pode ordenar o atendimento pelo score estimado de risco, combinando o nível de risco com a capacidade real de contato.",
            "- A existência de falsos positivos (clientes sinalizados que não cancelaram) e falsos negativos (165 clientes que cancelaram e ficaram abaixo do corte `0,50`) reforça que o score é um apoio à priorização, não uma decisão automática infalível.",
            "- A tabela completa dos 1.409 clientes ranqueados está exportada em [`reports/priority_customers_test.csv`](priority_customers_test.csv).",
            "",
        ]
    )

    return "\n".join(linhas)


def exportar_relatorio_prioritarios(
    caminho_dados: Path = ARQUIVO_DADOS_BRUTOS,
    caminho_modelo: Path = ARQUIVO_MODELO,
    caminho_csv: Path = ARQUIVO_FILA_PRIORITARIA,
    caminho_markdown: Path = ARQUIVO_RELATORIO_RISCO,
) -> dict:
    """Gera e exporta a fila de clientes prioritarios no conjunto de teste."""
    dados = carregar_dataset(caminho_dados)
    modelo = carregar_modelo(caminho_modelo)
    X, y = separar_features_e_alvo(dados)
    _, X_test, _, y_test = dividir_treino_e_teste(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    fila = gerar_fila_priorizacao(modelo, dados, X_test, y_real=y_test)
    resumo = resumir_fila_risco(fila)

    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    fila.to_csv(caminho_csv, index=False)

    conteudo_md = gerar_relatorio_risco_markdown(fila, resumo)
    caminho_markdown.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_markdown, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo_md)

    return {
        "fila": fila,
        "resumo": resumo,
    }


if __name__ == "__main__":
    saida = exportar_relatorio_prioritarios()
    resumo = saida["resumo"]
    print(
        f"Fila gerada para {resumo['total_avaliados']} clientes de teste. "
        f"Sinalizados (Yes): {resumo['total_sinalizados_yes']} "
        f"(receita mensal associada: US$ {resumo['receita_mensal_associada_sinalizados']:,.2f})."
    )
    print(f"CSV salvo em: {ARQUIVO_FILA_PRIORITARIA}")
    print(f"Relatorio salvo em: {ARQUIVO_RELATORIO_RISCO}")
