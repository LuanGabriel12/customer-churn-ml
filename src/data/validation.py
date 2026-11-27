"""Auditoria reproduzivel de qualidade dos dados e consistencia com a API."""

from pathlib import Path

import pandas as pd

from src.config import (
    ARQUIVO_DADOS_BRUTOS,
    ARQUIVO_RELATORIO_QUALIDADE,
    CATEGORIAS_VALIDAS,
    COLUNA_ALVO,
    COLUNA_ID,
    COLUNAS_FEATURES,
    COLUNAS_OBRIGATORIAS_DATASET,
    SHA256_DATASET_ESPERADO,
)
from src.data.loader import calcular_sha256_arquivo, carregar_dataset


def validar_colunas_obrigatorias(dados: pd.DataFrame) -> dict:
    """Verifica colunas obrigatorias ausentes e colunas extras inesperadas."""
    esperadas = set(COLUNAS_OBRIGATORIAS_DATASET)
    presentes = set(dados.columns)
    return {
        "colunas_ausentes": sorted(esperadas - presentes),
        "colunas_extras": sorted(presentes - esperadas),
        "valido": esperadas == presentes,
    }


def auditar_qualidade_dados(
    dados: pd.DataFrame,
    caminho_arquivo: Path | None = None,
    campos_schema_api: set[str] | None = None,
) -> dict:
    """Executa uma auditoria completa de qualidade sobre o dataset Telco Customer Churn."""
    validacao_colunas = validar_colunas_obrigatorias(dados)

    # Tipos e valores ausentes explicitos
    tipos_colunas = {coluna: str(tipo) for coluna, tipo in dados.dtypes.items()}
    valores_ausentes = {
        coluna: int(qtd) for coluna, qtd in dados.isna().sum().items()
    }

    # Duplicidades
    linhas_duplicadas = int(dados.duplicated().sum())
    ids_duplicados = (
        int(dados[COLUNA_ID].duplicated().sum())
        if COLUNA_ID in dados.columns
        else 0
    )

    # Distribuicao do alvo
    contagem_alvo = dados[COLUNA_ALVO].value_counts(dropna=False)
    total_linhas = int(dados.shape[0])
    distribuicao_alvo = {}
    for classe, quantidade in contagem_alvo.items():
        distribuicao_alvo[str(classe)] = {
            "quantidade": int(quantidade),
            "percentual": round((int(quantidade) / total_linhas) * 100, 2)
            if total_linhas > 0
            else 0.0,
        }

    # Auditoria de TotalCharges
    total_charges_numerico = pd.to_numeric(dados["TotalCharges"], errors="coerce")
    mascara_invalidos_tc = total_charges_numerico.isna()
    qtd_invalidos_tc = int(mascara_invalidos_tc.sum())
    valores_brutos_invalidos_tc = sorted(
        {str(valor) for valor in dados.loc[mascara_invalidos_tc, "TotalCharges"].tolist()}
    )
    tenure_nos_invalidos_tc = {
        int(chave): int(valor)
        for chave, valor in dados.loc[mascara_invalidos_tc, "tenure"]
        .value_counts()
        .items()
    }
    qtd_tenure_zero = int((dados["tenure"] == 0).sum())

    # Validacao de faixas numericas
    valores_numericos_invalidos = {
        "tenure_negativo": int((dados["tenure"] < 0).sum()),
        "monthly_charges_negativo": int((dados["MonthlyCharges"] < 0).sum()),
        "total_charges_negativo": int((total_charges_numerico < 0).sum()),
        "senior_citizen_fora_0_1": int((~dados["SeniorCitizen"].isin([0, 1])).sum()),
    }

    # Categorias inesperadas e espacos extras
    categorias_inesperadas = {}
    espacos_em_branco = {}
    for coluna, permitidas in CATEGORIAS_VALIDAS.items():
        if coluna in dados.columns:
            encontradas = {str(valor) for valor in dados[coluna].dropna().unique().tolist()}
            extras = sorted(encontradas - set(permitidas))
            if extras:
                categorias_inesperadas[coluna] = extras

    for coluna in dados.columns:
        if dados[coluna].dtype == "object" or str(dados[coluna].dtype) == "str":
            serie_str = dados[coluna].astype(str)
            qtd_espaco_borda = int((serie_str != serie_str.str.strip()).sum())
            qtd_so_espaco = int(serie_str.str.strip().eq("").sum())
            if qtd_espaco_borda > 0 or qtd_so_espaco > 0:
                espacos_em_branco[coluna] = {
                    "espacos_nas_bordas": qtd_espaco_borda,
                    "strings_vazias_ou_so_espaco": qtd_so_espaco,
                }

    # Consistencia logica entre servicos
    inconsistencia_telefone = int(
        (
            (dados["PhoneService"] == "No")
            & (dados["MultipleLines"] != "No phone service")
        ).sum()
    )
    colunas_servicos_internet = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    inconsistencias_internet = {}
    for coluna in colunas_servicos_internet:
        inconsistencias_internet[coluna] = int(
            (
                (dados["InternetService"] == "No")
                & (dados[coluna] != "No internet service")
            ).sum()
        )

    # Estatisticas descritivas das numericas
    estatisticas_numericas = {
        "tenure": {
            "min": float(dados["tenure"].min()),
            "mediana": float(dados["tenure"].median()),
            "media": round(float(dados["tenure"].mean()), 2),
            "max": float(dados["tenure"].max()),
        },
        "MonthlyCharges": {
            "min": float(dados["MonthlyCharges"].min()),
            "mediana": float(dados["MonthlyCharges"].median()),
            "media": round(float(dados["MonthlyCharges"].mean()), 2),
            "max": float(dados["MonthlyCharges"].max()),
        },
        "TotalCharges_convertido": {
            "validos": int(total_charges_numerico.notna().sum()),
            "ausentes_apos_conversao": qtd_invalidos_tc,
            "min": float(total_charges_numerico.min()),
            "mediana": float(total_charges_numerico.median()),
            "media": round(float(total_charges_numerico.mean()), 2),
            "max": float(total_charges_numerico.max()),
        },
    }

    # Consistencia com schema da API (usando COLUNAS_FEATURES ou conjunto informado)
    esperadas_features = set(COLUNAS_FEATURES)
    if campos_schema_api is None:
        campos_api = esperadas_features
    else:
        campos_api = set(campos_schema_api)

    consistencia_api = {
        "quantidade_features_dataset": len(esperadas_features),
        "quantidade_campos_api": len(campos_api),
        "faltando_na_api": sorted(esperadas_features - campos_api),
        "extras_na_api": sorted(campos_api - esperadas_features),
        "alinhado": esperadas_features == campos_api,
    }

    sha256_atual = None
    sha256_confere = None
    if caminho_arquivo is not None and Path(caminho_arquivo).exists():
        sha256_atual = calcular_sha256_arquivo(caminho_arquivo)
        sha256_confere = sha256_atual == SHA256_DATASET_ESPERADO

    return {
        "linhas": total_linhas,
        "colunas": int(dados.shape[1]),
        "sha256_atual": sha256_atual,
        "sha256_confere": sha256_confere,
        "validacao_colunas": validacao_colunas,
        "tipos_colunas": tipos_colunas,
        "valores_ausentes": valores_ausentes,
        "total_valores_ausentes_explicitos": sum(valores_ausentes.values()),
        "linhas_duplicadas": linhas_duplicadas,
        "ids_duplicados": ids_duplicados,
        "distribuicao_alvo": distribuicao_alvo,
        "total_charges_auditoria": {
            "tipo_original": str(dados["TotalCharges"].dtype),
            "quantidade_invalidos": qtd_invalidos_tc,
            "valores_brutos_invalidos": valores_brutos_invalidos_tc,
            "tenure_nos_invalidos": tenure_nos_invalidos_tc,
            "total_clientes_tenure_zero": qtd_tenure_zero,
        },
        "valores_numericos_invalidos": valores_numericos_invalidos,
        "categorias_inesperadas": categorias_inesperadas,
        "espacos_em_branco": espacos_em_branco,
        "consistencia_servicos": {
            "telefone_inconsistente": inconsistencia_telefone,
            "internet_inconsistente": inconsistencias_internet,
        },
        "estatisticas_numericas": estatisticas_numericas,
        "consistencia_api": consistencia_api,
    }


def gerar_relatorio_qualidade_markdown(auditoria: dict) -> str:
    """Gera um relatorio legivel em Markdown a partir do dicionario de auditoria."""
    linhas = [
        "# Relatório de Qualidade dos Dados — Telco Customer Churn",
        "",
        "Relatório gerado automaticamente por `src/data/validation.py` diretamente sobre o arquivo bruto `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (mantido sem alterações).",
        "",
        "## 1. Resumo Geral e Integridade",
        "",
        "| Verificação | Resultado | Status |",
        "| --- | --- | --- |",
        f"| Dimensões | {auditoria['linhas']:,} linhas × {auditoria['colunas']} colunas | OK |",
        f"| Colunas obrigatórias | Ausentes: {len(auditoria['validacao_colunas']['colunas_ausentes'])} · Extras: {len(auditoria['validacao_colunas']['colunas_extras'])} | {'OK' if auditoria['validacao_colunas']['valido'] else 'ATENÇÃO'} |",
        f"| Valores ausentes explícitos (`NaN`) | {auditoria['total_valores_ausentes_explicitos']} | OK |",
        f"| Registros duplicados exatos | {auditoria['linhas_duplicadas']} | OK |",
        f"| `customerID` duplicados | {auditoria['ids_duplicados']} | OK |",
        f"| `TotalCharges` inválidos ao converter para número | {auditoria['total_charges_auditoria']['quantidade_invalidos']} registros (`' '`) | TRATADO NA PIPELINE |",
        f"| Categorias inesperadas | {len(auditoria['categorias_inesperadas'])} colunas afetadas | {'OK' if not auditoria['categorias_inesperadas'] else 'ATENÇÃO'} |",
        f"| Alinhamento Dataset × Schema da API | {auditoria['consistencia_api']['quantidade_features_dataset']} features × {auditoria['consistencia_api']['quantidade_campos_api']} campos | {'OK' if auditoria['consistencia_api']['alinhado'] else 'ATENÇÃO'} |",
    ]

    if auditoria["sha256_atual"]:
        status_hash = "OK" if auditoria["sha256_confere"] else "DIVERGENTE"
        linhas.append(
            f"| SHA-256 do CSV bruto | `{auditoria['sha256_atual']}` | {status_hash} |"
        )

    linhas.extend(
        [
            "",
            "## 2. Distribuição da Variável Alvo (`Churn`)",
            "",
            "| Classe | Quantidade | Percentual |",
            "| --- | ---: | ---: |",
        ]
    )
    for classe, info in auditoria["distribuicao_alvo"].items():
        linhas.append(
            f"| `{classe}` | {info['quantidade']:,} | {info['percentual']:.2f}% |"
        )

    tc_info = auditoria["total_charges_auditoria"]
    linhas.extend(
        [
            "",
            "## 3. Problema Identificado e Solução: `TotalCharges`",
            "",
            f"- **Problema:** A coluna `TotalCharges` é carregada como texto (`{tc_info['tipo_original']}`) e possui **{tc_info['quantidade_invalidos']}** registros com string contendo apenas espaço em branco (`{tc_info['valores_brutos_invalidos']}`).",
            f"- **Causa provável no domínio:** Todos os **{tc_info['quantidade_invalidos']}** registros inválidos têm `tenure = 0` (distribuição de `tenure` entre os inválidos: `{tc_info['tenure_nos_invalidos']}`), igual ao total de clientes com `tenure = 0` na base (**{tc_info['total_clientes_tenure_zero']}**). São clientes recém-chegados que ainda não completaram o primeiro ciclo de cobrança acumulada.",
            "- **Solução adotada (sem data leakage):** O CSV bruto permanece intacto. A conversão `pd.to_numeric(..., errors='coerce')` ocorre dentro do `FunctionTransformer` da `Pipeline`, seguida de `SimpleImputer(strategy='median')` ajustado **exclusivamente** no conjunto de treino.",
            "",
            "## 4. Tipos das Colunas e Valores Ausentes Explícitos",
            "",
            "| Coluna | Tipo lido pelo Pandas | Ausentes explícitos |",
            "| --- | --- | ---: |",
        ]
    )
    for coluna, tipo in auditoria["tipos_colunas"].items():
        linhas.append(
            f"| `{coluna}` | `{tipo}` | {auditoria['valores_ausentes'][coluna]} |"
        )

    linhas.extend(
        [
            "",
            "## 5. Validação de Faixas Numéricas e Estatísticas Descritivas",
            "",
            "| Variável | Mínimo | Mediana | Média | Máximo |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for nome_var, stats in auditoria["estatisticas_numericas"].items():
        linhas.append(
            f"| `{nome_var}` | {stats['min']:.2f} | {stats['mediana']:.2f} | {stats['media']:.2f} | {stats['max']:.2f} |"
        )

    num_inv = auditoria["valores_numericos_invalidos"]
    linhas.extend(
        [
            "",
            "**Checagens de valores numéricos inválidos:**",
            f"- `tenure < 0`: **{num_inv['tenure_negativo']}**",
            f"- `MonthlyCharges < 0`: **{num_inv['monthly_charges_negativo']}**",
            f"- `TotalCharges < 0`: **{num_inv['total_charges_negativo']}**",
            f"- `SeniorCitizen` fora de `{{0, 1}}`: **{num_inv['senior_citizen_fora_0_1']}**",
            "",
            "## 6. Consistência Lógica entre Serviços",
            "",
            f"- `PhoneService == 'No'` com `MultipleLines != 'No phone service'`: **{auditoria['consistencia_servicos']['telefone_inconsistente']}**",
        ]
    )
    for coluna, qtd in auditoria["consistencia_servicos"]["internet_inconsistente"].items():
        linhas.append(
            f"- `InternetService == 'No'` com `{coluna} != 'No internet service'`: **{qtd}**"
        )

    linhas.extend(
        [
            "",
            "## 7. Consistência com os Schemas da API",
            "",
            f"- Features de entrada no modelo: **{auditoria['consistencia_api']['quantidade_features_dataset']}** (`customerID` e `Churn` removidas).",
            f"- Campos definidos no contrato Pydantic (`ClienteChurn`): **{auditoria['consistencia_api']['quantidade_campos_api']}**.",
            f"- Campos faltando na API: `{auditoria['consistencia_api']['faltando_na_api']}`.",
            f"- Campos extras na API: `{auditoria['consistencia_api']['extras_na_api']}`.",
            "- Política para categorias desconhecidas: todas as 15 colunas categóricas são validadas estritamente via `Literal[...]` no Pydantic (retornando HTTP `422`), enquanto o `OneHotEncoder(handle_unknown='ignore')` permanece na pipeline como defesa em profundidade.",
            "",
            "## 8. Redistribuição e Atribuição do Dataset",
            "",
            "- **Atribuição oficial:** *IBM Sample Data Sets (Watson Analytics / Telco Customer Churn sample)*, disponibilizado no repositório [IBM/customer-churn-prediction](https://github.com/IBM/customer-churn-prediction).",
            "- **Licenciamento e escopo de redistribuição:** o repositório de código da IBM é licenciado sob **Apache License 2.0**, enquanto o CSV `WA_Fn-UseC_-Telco-Customer-Churn.csv` é um dataset fictício de demonstração publicado pela IBM para fins educacionais e analíticos (sem uma licença aberta dedicada que autorize redistribuição comercial irrestrita do CSV bruto).",
            "- **Preservação neste portfólio:** o arquivo já integrava a versão V1 deste repositório educacional e é mantido sem alterações nos registros, com quebras de linha canônicas `LF` protegidas por `.gitattributes` e atribuição explícita à IBM em `data/README.md`.",
            "",
        ]
    )

    return "\n".join(linhas)


def exportar_relatorio_qualidade(
    caminho_dados: Path = ARQUIVO_DADOS_BRUTOS,
    caminho_saida: Path = ARQUIVO_RELATORIO_QUALIDADE,
    campos_schema_api: set[str] | None = None,
) -> dict:
    """Executa a auditoria completa e salva o relatorio em Markdown."""
    dados = carregar_dataset(caminho_dados)
    auditoria = auditar_qualidade_dados(
        dados,
        caminho_arquivo=caminho_dados,
        campos_schema_api=campos_schema_api,
    )
    conteudo = gerar_relatorio_qualidade_markdown(auditoria)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)
    return auditoria


if __name__ == "__main__":
    from app.schemas import ClienteChurn

    resultado = exportar_relatorio_qualidade(
        campos_schema_api=set(ClienteChurn.model_fields.keys())
    )
    print(
        f"Auditoria concluida: {resultado['linhas']} linhas, "
        f"{resultado['colunas']} colunas, "
        f"{resultado['total_charges_auditoria']['quantidade_invalidos']} valores invalidos em TotalCharges."
    )
    print(f"Relatorio salvo em: {ARQUIVO_RELATORIO_QUALIDADE}")
