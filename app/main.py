"""API FastAPI para previsao individual, em lote e inspecao do modelo de churn."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.schemas import (
    ClienteChurn,
    InfoModelo,
    LoteClientesChurn,
    RespostaChurn,
    RespostaLoteChurn,
    RespostaSaude,
)
from src.config import (
    ARQUIVO_METADADOS_MODELO,
    ARQUIVO_MODELO,
    CLASSE_POSITIVA,
    LIMITE_MAXIMO_LOTE,
)
from src.data.loader import calcular_sha256_arquivo
from src.modeling.predict import (
    carregar_metadados_modelo,
    carregar_modelo,
    obter_colunas_entrada_modelo,
    prever_cliente,
    prever_lote,
)

app = FastAPI(
    title="Customer Churn Intelligence API",
    description=(
        "API de demonstração para estimativa de risco de cancelamento de clientes "
        "(dataset IBM Telco Customer Churn) utilizando a pipeline documentada de Logistic Regression."
    ),
    version="2.0.0",
)

# Estado simples em modulo para armazenar a pipeline carregada e seus metadados verificaveis
ESTADO_APLICACAO = {
    "modelo": None,
    "metadados": None,
    "erro_carregamento": None,
}


def validar_correspondencia_modelo_metadados(
    modelo, metadados: dict, caminho_modelo: Path
) -> None:
    """Confere se o artefato serializado corresponde exatamente aos metadados carregados."""
    etapas_reais = (
        list(modelo.named_steps.keys()) if hasattr(modelo, "named_steps") else []
    )
    classes_reais = [str(classe) for classe in list(modelo.classes_)]
    features_reais = obter_colunas_entrada_modelo(modelo)
    algoritmo_real = (
        type(modelo.named_steps["modelo"]).__name__
        if hasattr(modelo, "named_steps") and "modelo" in modelo.named_steps
        else type(modelo).__name__
    )

    if metadados.get("algoritmo") != algoritmo_real:
        raise ValueError("Divergencia entre algoritmo do modelo e metadados.")
    if metadados.get("etapas_pipeline") != etapas_reais:
        raise ValueError("Divergencia entre etapas da pipeline e metadados.")
    if metadados.get("classes_modelo") != classes_reais:
        raise ValueError("Divergencia entre classes do modelo e metadados.")
    if metadados.get("features_esperadas") != features_reais:
        raise ValueError("Divergencia entre features esperadas e metadados.")

    sha256_esperado = metadados.get("modelo_sha256")
    if sha256_esperado:
        sha256_real = calcular_sha256_arquivo(caminho_modelo)
        if sha256_real != sha256_esperado:
            raise ValueError("Divergencia de integridade SHA-256 entre modelo e metadados.")


def inicializar_modelo(
    caminho_modelo: Path = ARQUIVO_MODELO,
    caminho_metadados: Path = ARQUIVO_METADADOS_MODELO,
) -> bool:
    """Carrega o modelo treinado e seus metadados validando correspondencia entre ambos."""
    try:
        modelo = carregar_modelo(caminho_modelo)
        metadados = (
            carregar_metadados_modelo(caminho_metadados)
            if Path(caminho_metadados).exists()
            else None
        )
        if metadados is not None:
            validar_correspondencia_modelo_metadados(modelo, metadados, caminho_modelo)

        ESTADO_APLICACAO["modelo"] = modelo
        ESTADO_APLICACAO["metadados"] = metadados
        ESTADO_APLICACAO["erro_carregamento"] = None
        return True
    except Exception:
        ESTADO_APLICACAO["modelo"] = None
        ESTADO_APLICACAO["metadados"] = None
        ESTADO_APLICACAO["erro_carregamento"] = "Falha ao carregar ou validar o modelo."
        return False


def obter_modelo_carregado():
    """Retorna o modelo em memoria ou responde com HTTP 503 sem expor caminhos internos."""
    modelo = ESTADO_APLICACAO["modelo"]
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo indisponivel no momento. Execute o fluxo de treinamento "
                "('python -m src.modeling.train') e reinicie a API."
            ),
        )
    return modelo


# Inicializa o modelo a partir do caminho padrao reproduzivel do projeto
inicializar_modelo()


@app.get(
    "/health",
    response_model=RespostaSaude,
    tags=["Operação"],
    summary="Verificar saúde da API e do modelo",
)
def verificar_saude():
    """Verifica a disponibilidade da aplicação e se o modelo está carregado."""
    if ESTADO_APLICACAO["modelo"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Aplicacao ativa, mas o modelo nao esta carregado. "
                "Execute 'python -m src.modeling.train' para gerar o artefato."
            ),
        )
    return {
        "status": "ok",
        "modelo_carregado": True,
        "detalhe": "Aplicacao e modelo prontos para previsao.",
    }


@app.get(
    "/model-info",
    response_model=InfoModelo,
    tags=["Operação"],
    summary="Consultar metadados verificáveis do modelo",
)
def obter_informacoes_modelo():
    """Retorna informações verificáveis sobre o modelo carregado e suas métricas."""
    modelo = obter_modelo_carregado()
    metadados = ESTADO_APLICACAO["metadados"]
    if metadados is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Metadados do modelo indisponiveis. "
                "Execute 'python -m src.modeling.train' para regenerar os artefatos."
            ),
        )

    classes_carregadas = [str(classe) for classe in list(modelo.classes_)]
    etapas_carregadas = (
        list(modelo.named_steps.keys()) if hasattr(modelo, "named_steps") else []
    )

    return {
        "nome_modelo": metadados["nome_modelo"],
        "algoritmo": metadados["algoritmo"],
        "versao_projeto": metadados["versao_projeto"],
        "treinado_em_utc": metadados["treinado_em_utc"],
        "etapas_pipeline": etapas_carregadas,
        "classes_modelo": classes_carregadas,
        "classe_positiva": metadados.get("classe_positiva", CLASSE_POSITIVA),
        "quantidade_features": len(obter_colunas_entrada_modelo(modelo)),
        "features_esperadas": obter_colunas_entrada_modelo(modelo),
        "registros_treino": metadados["registros_treino"],
        "registros_teste": metadados["registros_teste"],
        "dataset_sha256": metadados["dataset_sha256"],
        "modelo_sha256": metadados.get("modelo_sha256", ""),
        "limite_maximo_lote": LIMITE_MAXIMO_LOTE,
        "politica_categorias_desconhecidas": metadados[
            "politica_categorias_desconhecidas"
        ],
        "metricas_validacao_cruzada": metadados["metricas_validacao_cruzada"],
        "metricas_teste_historico": metadados["metricas_teste_historico"],
    }


@app.post(
    "/predict",
    response_model=RespostaChurn,
    tags=["Previsão"],
    summary="Previsão individual de churn",
)
def predicao_individual(cliente: ClienteChurn):
    """Realiza a previsão de risco de churn para um único cliente."""
    modelo = obter_modelo_carregado()
    return prever_cliente(modelo, cliente.model_dump())


@app.post(
    "/predict/batch",
    response_model=RespostaLoteChurn,
    tags=["Previsão"],
    summary="Previsão em lote de churn",
)
def predicao_em_lote(lote: LoteClientesChurn):
    """Realiza previsões de risco de churn para múltiplos clientes em uma requisição."""
    modelo = obter_modelo_carregado()
    registros = [cliente.model_dump() for cliente in lote.clientes]
    previsoes = prever_lote(modelo, registros)
    return {
        "total_clientes": len(previsoes),
        "previsoes": previsoes,
    }
