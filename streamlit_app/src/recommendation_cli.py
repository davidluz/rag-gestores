# recommendation_cli.py
# CLI autônomo: gera devolutiva e recomendações sem Streamlit

import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# === Configurações de caminhos ===
APP_ROOT = Path(__file__).resolve().parent.parent
# repita o mesmo índice/pickle 1606 usados no Streamlit:
INDEX_PATH       = APP_ROOT / "models" / "odas_index_1606.faiss"
META_ODAS_PATH   = APP_ROOT / "models" / "metadados_odas_1606.pkl"
DEVOLUTIVAS_PATH = APP_ROOT / "data"   / "devolutivas.csv"
RUBRICAS_PATH    = APP_ROOT / "data"   / "rubricas.csv"

# === Carregamento de dados ===

def load_model() -> SentenceTransformer:
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def load_index(path: Path) -> faiss.Index:
    idx = faiss.read_index(str(path))
    print(f"DEBUG: índice FAISS carregado de '{path}', d = {idx.d}")
    return idx


def load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=';')

# === Lógica de devolutiva dinâmica ===

def encontrar_rubrica(df_rub, pontuacao, dimensao, subdimensao):
    cand = df_rub[
        (df_rub['dimensao']==dimensao)&
        (df_rub['subdimensao']==subdimensao)&
        (df_rub['faixa_total_min']<=pontuacao)&
        (df_rub['faixa_total_max']>=pontuacao)
    ]
    if cand.empty: return None, None, None
    nro  = int(cand.iloc[0]['rubrica_numero'])
    nome = cand.iloc[0]['rubrica_nome']
    faixa = cand[(cand['subfaixa_min']<=pontuacao)&(cand['subfaixa_max']>=pontuacao)]
    tipo = faixa.iloc[0]['tipo_faixa'] if not faixa.empty else None
    return nro, nome, tipo


def formatar_necessidades(texto: str) -> str:
    if not isinstance(texto, str) or not texto.strip(): return "Sem necessidades formativas informadas."
    md = ""
    for linha in texto.split("\n"):
        partes = [p.strip() for p in linha.split("•") if p.strip()]
        if not partes: continue
        md += f"- **{partes[0]}**\n"
        for sub in partes[1:]: md += f"  - {sub}\n"
    return md.strip()


def gerar_devolutiva(df_dev, df_rub, pont, dim, sub):
    nro, nome, tipo = encontrar_rubrica(df_rub, pont, dim, sub)
    if not nome or not tipo: return None
    sel = df_dev[
        (df_dev['Dimensão']==dim)&
        (df_dev['Subdimensão']==sub)&
        (df_dev['Rubrica numero']==nro)&
        (df_dev['Rubrica nome']==f"{nome} – Nível {tipo}")
    ]
    if sel.empty: return None
    item = sel.iloc[0]
    texto = (
        f"Pontuação: {pont}\n"
        f"Dimensão: {dim}\n"
        f"Subdimensão: {sub}\n"
        f"Rubrica: {nome} – Nível {tipo}\n"
        f"\nNecessidades formativas:\n{item['Necessidades formativas']}"
    )
    return texto

# === Pipeline de recomendações simples ===

def gerar_embedding(modelo, texto: str) -> np.ndarray:
    emb = modelo.encode([texto])
    embn = emb/np.linalg.norm(emb,axis=1,keepdims=True)
    return embn.astype('float32')


def get_simple_recommendations(modelo, index, df_odas, texto_q):
    # busca vetorial
    emb = gerar_embedding(modelo, texto_q)
    dist, idxs = index.search(emb, 1000)
    df = df_odas.iloc[idxs[0]].copy()
    df['distância'] = dist[0]
    # categorias
    regs = {
        'interativos': r'jogo|painel',
        'visuais':     r'infográfico|mapa|tabela|gráfico|slide',
        'videos':      r'vídeo|video|curso|aula|aula gravada|palestra|webinário|animação|exposição',
        'audios':      r'áudio|audio|podcast|rádio|entrevista',
        'artigos':     r'texto|artigo|livro|relatório|resenha|plano de aula|documento institucional|manual|guia|tutorial|documento oficial|documento técnico|cartilha|blog|apostila|coletânea|recomendação'
    }
    conds = [df['Suporte'].str.contains(rx, case=False, na=False) for rx in regs.values()]
    cats = list(regs.keys())
    df['categoria'] = np.select(conds,cats,default='outros')
    top = df[df['categoria']!='outros'].groupby('categoria').head(10)
    return {cat: top[top['categoria']==cat] for cat in cats}

# === Execução CLI ===
if __name__=='__main__':
    # Entradas do usuário (modificar aqui)
    pontuacao   = 1
    dimensao    = 'Dimensão pedagógica'
    subdimensao = 'Planejamento pedagógico'

    # Carrega dados
    modelo    = load_model()
    index     = load_index(INDEX_PATH)
    df_odas   = load_pickle(META_ODAS_PATH)
    df_dev    = load_csv(DEVOLUTIVAS_PATH)
    df_rub    = load_csv(RUBRICAS_PATH)

    # Gera texto rico automaticamente
    texto_rico = gerar_devolutiva(df_dev, df_rub, pontuacao, dimensao, subdimensao)
    print("=== Devolutiva Gerada ===")
    print(texto_rico or "(não foi possível gerar devolutiva)")

    # Recomendação simples
    recs = get_simple_recommendations(modelo,index,df_odas,texto_rico or "")
    print("\n=== Recomendações ===")
    for cat, df_cat in recs.items():
        print(f"\n-- {cat.upper()} --")
        if df_cat.empty:
            print("  (sem resultados)")
        else:
            for _, row in df_cat.iterrows():
                print(f"  * {row['Título']} ({row['distância']:.4f}) - {row['Fonte']}")

