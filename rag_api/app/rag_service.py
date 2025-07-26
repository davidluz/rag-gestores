"""
Camada de serviço: carrega modelos/pickles uma única vez e expõe funções
que são chamadas pelo endpoint.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .settings import settings
from .models import ODAItem


# ------------------------
# Carregamento de recursos
# ------------------------

MODEL: SentenceTransformer | None = None
INDEX: faiss.Index          | None = None
DF_ODAS: pd.DataFrame       | None = None
DF_DEV: pd.DataFrame        | None = None
DF_RUB: pd.DataFrame        | None = None


def _load_model() -> SentenceTransformer:
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def _load_index(path: Path) -> faiss.Index:
    idx = faiss.read_index(str(path))
    return idx


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def init() -> None:
    """Chame uma vez (startup)."""
    global MODEL, INDEX, DF_ODAS, DF_DEV, DF_RUB

    if MODEL is not None:
        return  # já carregado

    MODEL   = _load_model()
    INDEX   = _load_index(settings.index_path)
    DF_ODAS = _load_pickle(settings.meta_odas_path)
    DF_DEV  = _load_csv(settings.devolutivas_path)
    DF_RUB  = _load_csv(settings.rubricas_path)


# ------------------------
#   LÓGICA de NEGÓCIO
# ------------------------

def _encontrar_rubrica(df_rub, pontuacao, dimensao, subdimensao) -> Tuple[int | None, str | None, str | None]:
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
    md = []
    for linha in texto.split("\n"):
        partes = [p.strip() for p in linha.split("•") if p.strip()]
        if not partes:
            continue
        md.append(f"- **{partes[0]}**")
        for sub in partes[1:]:
            md.append(f"  - {sub}")
    return "\n".join(md)


def _gerar_devolutiva(df_dev, df_rub, pont, dim, sub) -> Tuple[str | None, int | None, str | None, str | None]:
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


def _gerar_embedding(modelo, texto: str) -> np.ndarray:
    emb = modelo.encode([texto])
    embn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return embn.astype("float32")


def _get_simple_recommendations(
    modelo, index, df_odas, texto_q
) -> Dict[str, List[ODAItem]]:
    # Busca vetorial
    emb = _gerar_embedding(modelo, texto_q)
    dist, idxs = index.search(emb, 1000)
    df = df_odas.iloc[idxs[0]].copy()
    df["distância"] = dist[0]

    # Categorias
    regs = {
        "interativos": r"jogo|painel",
        "visuais": r"infográfico|mapa|tabela|gráfico|slide",
        "videos": r"vídeo|video|curso|aula|aula gravada|palestra|webinário|animação|exposição",
        "audios": r"áudio|audio|podcast|rádio|entrevista",
        "artigos": r"texto|artigo|livro|relatório|resenha|plano de aula|documento institucional|manual|guia|tutorial|documento oficial|documento técnico|cartilha|blog|apostila|coletânea|recomendação",
    }
    conds = [df["Suporte"].str.contains(rx, case=False, na=False) for rx in regs.values()]
    cats = list(regs.keys())
    df["categoria"] = np.select(conds, cats, default="outros")
    top = df[df["categoria"] != "outros"].groupby("categoria").head(10)

    resultado: Dict[str, List[ODAItem]] = {}
    for cat in cats:
        df_cat = top[top["categoria"] == cat]
        resultado[cat] = [
            ODAItem(
                titulo=row["Título"],
                fonte=row.get("Fonte"),
                distancia=float(row["distância"]),
                url=row.get("URL"),
            )
            for _, row in df_cat.iterrows()
        ]
    return resultado


# ------------------------
#  Interface pública
# ------------------------

def recommend(pontuacao: int, dimensao: str, subdimensao: str):
    """Executa a pipeline completa e retorna dicionário serializável em JSON."""
    if MODEL is None:
        init()

    texto_rico, nro, nome, tipo = _gerar_devolutiva(
        DF_DEV, DF_RUB, pontuacao, dimensao, subdimensao
    )
    recs = _get_simple_recommendations(MODEL, INDEX, DF_ODAS, texto_rico or "")

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
