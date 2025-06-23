# scripts/train_model.py

import os
import pickle
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional

import faiss
import mlflow
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURAÇÃO DO EXPERIMENTO ---
MODELO_VERSAO = "v1_base" # Mude a versão a cada novo experimento
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 16

# --- 2. SETUP DE AMBIENTE E CAMINHOS ---
PROJECT_ROOT = Path(__file__).parent.parent
caminho_xlsx_input = PROJECT_ROOT / "data_source" / "Base_de_ODAS_1606.xlsx"
caminho_devolutivas_input = PROJECT_ROOT / "streamlit_app" / "data" / "devolutivas.csv"
pasta_output_models = PROJECT_ROOT / "streamlit_app" / "models" / MODELO_VERSAO
os.makedirs(pasta_output_models, exist_ok=True)
caminho_metadados_output = pasta_output_models / "metadados_odas.pkl"
caminho_index_output = pasta_output_models / "odas_index.faiss"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Dispositivo selecionado: {device.upper()}")

# --- 3. FUNÇÕES AUXILIARES ---
def preparar_texto_oda_enriquecido(row: pd.Series) -> str:
    header = f"Contexto: Público '{row.get('Público-Alvo', 'Geral')}', Rubrica '{row.get('Rubrica_IA', 'N/A')}'."
    partes = [header, f"Título: {row.get('Título', '')}", f"Resumo completo: {row.get('Resumo completo', '')}", f"Temas: {row.get('Temas', '')}"]
    return "\n".join([p for p in partes if pd.notnull(p) and str(p).strip() and 'nan' not in str(p).lower()])

def gerar_embedding(modelo: SentenceTransformer, textos: list) -> np.ndarray:
    embeddings = modelo.encode(textos, batch_size=BATCH_SIZE, show_progress_bar=True, device=device, trust_remote_code=True)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# --- 4. PIPELINE PRINCIPAL ---
def main():
    mlflow.set_experiment("Recomendacao_Mori_Pipeline")
    with mlflow.start_run(run_name=f"Treino_Versao_{MODELO_VERSAO}") as run:
        print(f"\n🚀 Iniciando Run: {MODELO_VERSAO} (ID: {run.info.run_id})")
        mlflow.log_params({"versao_modelo": MODELO_VERSAO, "embedding_model": EMBEDDING_MODEL, "batch_size": BATCH_SIZE})

        print("   - Carregando e processando dados...")
        df_odas = pd.read_excel(caminho_xlsx_input)
        df_odas.dropna(subset=["Resumo completo", "Temas", "Dimensões", "Tipo"], inplace=True)
        df_odas["texto_completo"] = df_odas.apply(preparar_texto_oda_enriquecido, axis=1)

        print("   - Gerando gráficos de análise...")
        sns.set_theme(style="whitegrid")
        fig_suporte, ax1 = plt.subplots(figsize=(10, 8)); df_odas['Suporte'].value_counts().head(15).sort_values().plot(kind='barh', ax=ax1); ax1.set_title('Top 15 Tipos de Suporte'); mlflow.log_figure(fig_suporte, "analise_exploratoria/1_distribuicao_suporte.png")
        if 'Rubrica_IA' in df_odas.columns:
            fig_rubrica, ax2 = plt.subplots(figsize=(10, 8)); df_odas['Rubrica_IA'].value_counts().plot(kind='pie', autopct='%1.1f%%'); ax2.set_title('Distribuição por Rubrica_IA'); ax2.set_ylabel(''); mlflow.log_figure(fig_rubrica, "analise_exploratoria/2_distribuicao_rubrica_ia.png")
        plt.close('all')

        print(f"   - Gerando embeddings com '{EMBEDDING_MODEL}'...")
        modelo = SentenceTransformer(EMBEDDING_MODEL, device=device, trust_remote_code=True)
        embeddings = gerar_embedding(modelo, df_odas["texto_completo"].tolist())
        
        dimensao = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimensao); index.add(embeddings.astype("float32"))
        
        print(f"   - Salvando artefatos na pasta: {pasta_output_models}...")
        metadados = df_odas[[c for c in df_odas.columns if c != 'texto_completo']].reset_index(drop=True)
        with open(caminho_metadados_output, "wb") as f: pickle.dump(metadados, f)
        faiss.write_index(index, str(caminho_index_output))
        mlflow.log_artifacts(str(pasta_output_models), artifact_path="model_artifacts")

        mlflow.log_metric("num_documentos_indexados", index.ntotal)
        mlflow.log_metric("dimensao_vetores", dimensao)

        print("\n" + "="*50)
        print(f"✅ Experimento {MODELO_VERSAO} finalizado com sucesso!")
        print("   Para visualizar, rode 'mlflow ui' no terminal da pasta raiz do projeto.")
        print("="*50)

if __name__ == "__main__":
    main()