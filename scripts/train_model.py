# scripts/train_model.py

import os
import pickle
import re
import time
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import mlflow
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURAÇÃO DO EXPERIMENTO ---
MODELO_VERSAO = "v6_processamento_em_memoria" # Nova versão para o log
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 32
CHUNK_SIZE = 1000 # Continuamos usando para fatiar o DataFrame

# --- 2. SETUP DE AMBIENTE E CAMINHOS ---
PROJECT_ROOT = Path.cwd()
caminho_xlsx_input = PROJECT_ROOT / "data_source" / "Base_de_ODAS_1606.xlsx"
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

# --- 4. PIPELINE PRINCIPAL COM PROCESSAMENTO EM LOTES ---
def main():
    """Função principal que orquestra o pipeline de treinamento em lotes."""
    
    mlflow.set_experiment("Recomendacao_Mori_Pipeline")
    with mlflow.start_run(run_name=f"Treino_Versao_{MODELO_VERSAO}") as run:
        print(f"\n🚀 Iniciando Run: {MODELO_VERSAO} (ID: {run.info.run_id})")
        mlflow.log_params({"versao_modelo": MODELO_VERSAO, "embedding_model": EMBEDDING_MODEL, "chunk_size": CHUNK_SIZE})

        # --- AQUI ESTÁ A MUDANÇA ---
        # Carregamos o DataFrame inteiro de uma vez para a memória.
        print(f"   - Carregando dados de {caminho_xlsx_input} para a memória...")
        df_odas = pd.read_excel(caminho_xlsx_input)
        df_odas.dropna(subset=["Resumo completo", "Temas", "Dimensões", "Tipo"], inplace=True)
        print(f"   - {len(df_odas)} documentos carregados e prontos para processar.")

        # O resto do pré-processamento acontece no DataFrame completo
        df_odas["texto_completo"] = df_odas.apply(preparar_texto_oda_enriquecido, axis=1)

        # Carrega o modelo de embedding uma única vez
        modelo = SentenceTransformer(EMBEDDING_MODEL, device=device, trust_remote_code=True)
        dimensao = modelo.get_sentence_embedding_dimension()
        mlflow.log_param("device", device)

        # Inicializa um índice FAISS vazio e uma lista para os metadados
        index = faiss.IndexFlatIP(dimensao)
        lista_metadados = []

        print("\nIniciando processamento de embeddings em lotes (chunks)...")
        # Agora, fatiamos o DataFrame que já está na memória
        for i in range(0, len(df_odas), CHUNK_SIZE):
            chunk_df = df_odas.iloc[i:i + CHUNK_SIZE]
            print(f"--- Processando Lote {i // CHUNK_SIZE + 1} (linhas {i} a {i + CHUNK_SIZE}) ---")

            if chunk_df.empty: continue

            # Gera embeddings apenas para o lote atual
            print(f"   - Gerando embeddings para {len(chunk_df)} documentos...")
            embeddings = modelo.encode(chunk_df["texto_completo"].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Adiciona os embeddings do lote ao índice principal
            index.add(embeddings_norm.astype("float32"))
            
            # Guarda os metadados do lote (sem a coluna de texto completo)
            colunas_para_salvar = [col for col in chunk_df.columns if col != 'texto_completo']
            lista_metadados.append(chunk_df[colunas_para_salvar])

        print("\nProcessamento de lotes finalizado.")
        
        # Concatena todos os metadados em um único DataFrame
        metadados_finais = pd.concat(lista_metadados, ignore_index=True).reset_index(drop=True)
        
        # Salvando os artefatos finais
        print(f"Salvando artefatos em {pasta_output_models}...")
        with open(caminho_metadados_output, "wb") as f: pickle.dump(metadados_finais, f)
        faiss.write_index(index, str(caminho_index_output))
        
        # Log no MLflow
        mlflow.log_artifacts(str(pasta_output_models), artifact_path="model_artifacts")
        mlflow.log_metric("num_documentos_indexados", index.ntotal)
        mlflow.log_metric("dimensao_vetores", dimensao)
        
        print("\n" + "="*50)
        print(f"✅ Experimento {MODELO_VERSAO} finalizado com sucesso!")
        print("="*50)

if __name__ == "__main__":
    main()