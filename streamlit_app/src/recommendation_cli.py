# recommendation_cli.py
# Versão CLI sem dependência de Streamlit, usando sempre busca simples

import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# === Configurações de caminhos ===
APP_ROOT = Path(__file__).resolve().parent.parent
# Ajuste para o mesmo índice e pickle usados no Streamlit (768-dimensions)
INDEX_PATH     = APP_ROOT / "models" / "odas_index_1606.faiss"
META_ODAS_PATH = APP_ROOT / "models" / "metadados_odas_1606.pkl"
RUBRICAS_PATH  = APP_ROOT / "data"   / "rubricas.csv"

# === Funções auxiliares ===

def load_model() -> SentenceTransformer:
    # Usa o mesmo modelo do Streamlit para gerar embeddings de 1606 dimensões
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def load_index(path: Path) -> faiss.Index:
    idx = faiss.read_index(str(path))
    print(f"DEBUG: Faiss index loaded from '{path}'. Expecting dimension d = {idx.d}")
    return idx


def load_metadata(path: Path) -> pd.DataFrame:
    with open(path, 'rb') as f:
        return pickle.load(f)


def gerar_embedding(modelo, texto: str) -> np.ndarray:
    emb = modelo.encode([texto])
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    print(f"DEBUG: Embedding generated. Shape before normalize = {emb.shape}, after normalize = {emb_norm.shape}")
    return emb_norm


def get_simple_recommendations(
    modelo,
    index,
    df_odas: pd.DataFrame,
    pontuacao: int,
    dimensao: str,
    subdimensao: str,
    texto_rico: str
):
    # Etapa 1: busca vetorial ampla
    emb_q = gerar_embedding(modelo, texto_rico).astype('float32')
    print(f"DEBUG: Query embedding for search. emb_q shape = {emb_q.shape}")
    dist, idxs = index.search(emb_q, 1000)
    resultados = df_odas.iloc[idxs[0]].copy()
    resultados['distância'] = dist[0]

    # Regex completo conforme original
    categorias_regex = {
        'interativos': r'jogo|painel',
        'visuais':     r'infográfico|mapa|tabela|gráfico|slide',
        'videos':      r'vídeo|video|curso|aula|aula gravada|palestra|webinário|animação|exposição',
        'audios':      r'áudio|audio|podcast|rádio|entrevista',
        'artigos':     r'texto|artigo|livro|relatório|resenha|plano de aula|documento institucional|manual|guia|tutorial|documento oficial|documento técnico|cartilha|blog|apostila|coletânea|recomendação'
    }

    # Categoriza e retém top 10 de cada
    conds = [resultados['Suporte'].str.contains(rx, case=False, na=False) for rx in categorias_regex.values()]
    cats = list(categorias_regex.keys())
    resultados['categoria'] = np.select(conds, cats, default='outros')
    top = resultados[resultados['categoria'] != 'outros'].groupby('categoria').head(10)
    return tuple(top[top['categoria'] == cat] for cat in ['artigos','videos','audios','visuais','interativos'])

# === Execução de exemplo ===
if __name__ == '__main__':
    # --- MOCK inputs ---
    pontuacao   = 35
    dimensao    = 'Conhecimento'
    subdimensao = 'Teoria'
    texto_rico  = 'Necessidades formativas: aprimorar compreensão teórica e aplicação prática.'

    # Carrega modelo, índice e dados
    modelo   = load_model()
    index    = load_index(INDEX_PATH)
    df_odas  = load_metadata(META_ODAS_PATH)

    # Busca simples (modelo intermediário)
    artigos, videos, audios, visuais, interativos = get_simple_recommendations(
        modelo, index, df_odas,
        pontuacao, dimensao, subdimensao, texto_rico
    )

    # Exibe no terminal
    categorias = ['Artigos','Vídeos','Áudios','Visuais','Interativos']
    for nome, df in zip(categorias, [artigos, videos, audios, visuais, interativos]):
        print(f"\n=== {nome} ===")
        if df.empty:
            print("(sem resultados)")
        else:
            for _, row in df.iterrows():
                titulo = row.get('Título','Sem título')
                fonte  = row.get('Fonte','#')
                score  = row.get('distância',0)
                print(f"- {titulo} [{fonte}] (score: {score:.4f})")
