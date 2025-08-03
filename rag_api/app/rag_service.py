"""
rag_service.py
Camada de serviço: carrega modelos/pickles uma única vez e expõe funções
que são chamadas pelo endpoint FastAPI.
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .settings import settings
from .models import ODAItem

# --------------------------------------------------------------------------- #
#  1.  Exceção própria para erros de validação                                #
# --------------------------------------------------------------------------- #
class RagValidationError(Exception):
    """Erro levantado quando o formato de entrada não corresponde ao esperado."""

    def __init__(self, campo: str, esperado: str, recebido: str):
        super().__init__(f"{campo}: esperado {esperado}, recebido {recebido}")
        self.campo = campo
        self.esperado = esperado
        self.recebido = recebido

    def to_dict(self) -> dict:
        return {
            "erro": "validacao",
            "campo": self.campo,
            "esperado": self.esperado,
            "recebido": self.recebido,
        }


# --------------------------------------------------------------------------- #
#  2.  Recursos carregados em cache                                           #
# --------------------------------------------------------------------------- #
MODEL: Optional[SentenceTransformer] = None
INDEX: Optional[faiss.Index] = None
DF_ODAS: Optional[pd.DataFrame] = None
DF_DEV: Optional[pd.DataFrame] = None
DF_RUB: Optional[pd.DataFrame] = None

# Lock para evitar condição de corrida em ambientes multi-thread / multi-worker
_INIT_LOCK = threading.Lock()


def _load_model() -> SentenceTransformer:
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def _load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def init() -> None:
    """Carrega modelos e datasets uma única vez (execução de startup)."""
    global MODEL, INDEX, DF_ODAS, DF_DEV, DF_RUB

    # Garante que apenas um thread/processo execute o bloco de carregamento
    with _INIT_LOCK:
        if MODEL is not None:
            return  # já carregado

        MODEL = _load_model()
        INDEX = _load_index(settings.index_path)          # StellaV5 final
        DF_ODAS = _load_pickle(settings.meta_odas_path)   # metadados com novas colunas
        DF_DEV = _load_csv(settings.devolutivas_path)
        DF_RUB = _load_csv(settings.rubricas_path)


# --------------------------------------------------------------------------- #
#  3.  Helpers de validação de entrada                                        #
# --------------------------------------------------------------------------- #
_CATS_VALIDAS = {"videos", "artigos", "audios", "visuais", "interativos"}


def _validar_entrada(pontuacao: int, dimensao: str, subdimensao: str) -> None:
    if not isinstance(pontuacao, int):
        raise RagValidationError("pontuacao", "int", f"{pontuacao} ({type(pontuacao).__name__})")
    if pontuacao < 0:
        raise RagValidationError("pontuacao", ">= 0", str(pontuacao))

    if not isinstance(dimensao, str) or not dimensao.strip():
        raise RagValidationError("dimensao", "str não vazia", repr(dimensao))

    if not isinstance(subdimensao, str) or not subdimensao.strip():
        raise RagValidationError("subdimensao", "str não vazia", repr(subdimensao))


def _validar_preferencias(preferencias: Optional[List[str]]) -> None:
    if preferencias is None:
        return
    if not isinstance(preferencias, list):
        raise RagValidationError("preferencias", "lista[str]", repr(preferencias))
    invalidas = [p for p in preferencias if p not in _CATS_VALIDAS]
    if invalidas:
        raise RagValidationError(
            "preferencias",
            f"subconjunto de {_CATS_VALIDAS}",
            f"{invalidas}",
        )


# --------------------------------------------------------------------------- #
#  4.  Lógica de negócio                                                      #
# --------------------------------------------------------------------------- #
def _encontrar_rubrica(
    df_rub: pd.DataFrame,
    pontuacao: int,
    dimensao: str,
    subdimensao: str,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    cand = df_rub[
        (df_rub["dimensao"] == dimensao)
        & (df_rub["subdimensao"] == subdimensao)
        & (df_rub["faixa_total_min"] <= pontuacao)
        & (df_rub["faixa_total_max"] >= pontuacao)
    ]
    if cand.empty:
        return None, None, None
    nro = int(cand.iloc[0]["rubrica_numero"])
    nome = cand.iloc[0]["rubrica_nome"]
    faixa = cand[
        (cand["subfaixa_min"] <= pontuacao) & (cand["subfaixa_max"] >= pontuacao)
    ]
    tipo = faixa.iloc[0]["tipo_faixa"] if not faixa.empty else None
    return nro, nome, tipo


def _formatar_necessidades(texto: str) -> str:
    if not isinstance(texto, str) or not texto.strip():
        return "Sem necessidades formativas informadas."
    md: list[str] = []
    for linha in texto.split("\n"):
        partes = [p.strip() for p in linha.split("•") if p.strip()]
        if not partes:
            continue
        md.append(f"- **{partes[0]}**")
        for sub in partes[1:]:
            md.append(f"  - {sub}")
    return "\n".join(md)


def _gerar_devolutiva(
    df_dev: pd.DataFrame,
    df_rub: pd.DataFrame,
    pont: int,
    dim: str,
    sub: str,
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    nro, nome, tipo = _encontrar_rubrica(df_rub, pont, dim, sub)
    if not nome or not tipo:
        return None, nro, nome, tipo

    sel = df_dev[
        (df_dev["Dimensão"] == dim)
        & (df_dev["Subdimensão"] == sub)
        & (df_dev["Rubrica numero"] == nro)
        & (df_dev["Rubrica nome"] == f"{nome} – Nível {tipo}")
    ]
    if sel.empty:
        return None, nro, nome, tipo

    item = sel.iloc[0]
    texto = (
        f"Pontuação: {pont}\n"
        f"Dimensão: {dim}\n"
        f"Subdimensão: {sub}\n"
        f"Rubrica: {nome} – Nível {tipo}\n"
        f"\nNecessidades formativas:\n{_formatar_necessidades(item['Necessidades formativas'])}"
    )
    return texto, nro, nome, tipo


def _gerar_embedding(modelo: SentenceTransformer, texto: str) -> np.ndarray:
    emb = modelo.encode([texto])
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1  # evita divisão por zero
    embn = emb / norm
    return embn.astype("float32")


def _safe_str(val: Any, default: str = "") -> str:
    """Converte NaN/None/vazio para string default; caso contrário, str(val)."""
    return default if pd.isna(val) or val == "" else str(val)


def _get_simple_recommendations(
    modelo: SentenceTransformer,
    index: faiss.Index,
    df_odas: pd.DataFrame,
    texto_q: str,
    preferencias: Optional[List[str]] = None,
) -> Dict[str, List[ODAItem]]:
    # Busca vetorial
    emb = _gerar_embedding(modelo, texto_q)
    dist, idxs = index.search(emb, 1000)

    # Remove índices inválidos (-1)
    mask_valid = idxs[0] >= 0
    if not mask_valid.any():
        return {}

    dist_valid = dist[0][mask_valid]
    idx_valid = idxs[0][mask_valid]

    df = df_odas.iloc[idx_valid].copy()
    df["distância"] = dist_valid

    # Categorias
    regs = {
        "interativos": r"jogo|painel",
        "visuais": r"infográfico|mapa|tabela|gráfico|slide",
        "videos": r"vídeo|video|curso|aula|aula gravada|palestra|webinário|animação|exposição",
        "audios": r"áudio|audio|podcast|rádio|entrevista",
        "artigos": r"texto|artigo|livro|relatório|resenha|plano de aula|documento institucional|manual|guia|tutorial|documento oficial|documento técnico|cartilha|blog|apostila|coletânea|recomendação",
    }
    cats = list(regs.keys())
    if preferencias:  # filtra/ordena conforme preferências
        cats = [c for c in cats if c in preferencias]

    conds = [df["Suporte"].str.contains(rx, case=False, na=False) for rx in regs.values()]
    df["categoria"] = np.select(conds, list(regs.keys()), default="outros")
    top = df[df["categoria"] != "outros"].groupby("categoria").head(10)

    resultado: Dict[str, List[ODAItem]] = {}
    for cat in cats:
        df_cat = top[top["categoria"] == cat]
        resultado[cat] = [
            ODAItem(
                titulo=_safe_str(row["Título"]),
                fonte=_safe_str(row.get("Fonte")),
                resumo=_safe_str(row.get("Resumo")),     #  ← NOVO
                distancia=float(row["distância"]),
                url=_safe_str(row.get("URL")),
            )
            for _, row in df_cat.iterrows()
        ]
    return resultado


# --------------------------------------------------------------------------- #
#  5.  Interface pública                                                      #
# --------------------------------------------------------------------------- #
def recommend(
    pontuacao: int,
    dimensao: str,
    subdimensao: str,
    preferencias: Optional[List[str]] = None,
) -> dict:
    """
    Executa a pipeline completa.
    • Em caso de erro de validação retorna {"erro": "..."}.
    • Em caso de sucesso retorna dicionário serializável em JSON.
    """
    try:
        _validar_entrada(pontuacao, dimensao, subdimensao)
        _validar_preferencias(preferencias)
    except RagValidationError as e:
        return e.to_dict()

    if MODEL is None:
        init()

    texto_rico, nro, nome, tipo = _gerar_devolutiva(
        DF_DEV, DF_RUB, pontuacao, dimensao, subdimensao
    )
    recs = _get_simple_recommendations(
        MODEL, INDEX, DF_ODAS, texto_rico or "", preferencias
    )

    return {
        "pontuacao": pontuacao,
        "dimensao": dimensao,
        "subdimensao": subdimensao,
        "rubrica_numero": nro,
        "rubrica_nome": nome,
        "rubrica_tipo": tipo,
        "devolutiva": texto_rico,
        "recomendacoes": recs,
    }
