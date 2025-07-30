# pages/3_Recomendação_Devolutivas_Padronizados.py

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

app_root = str(Path(__file__).parent.parent)
if app_root not in sys.path:
    sys.path.append(app_root)

from src.utils import (
    exibir_cabecalho,
    carregar_modelo_st,
    carregar_index,
    carregar_metadados,
    carregar_devolutivas_revisto,
    carregar_rubricas,
    gerar_texto_padronizado_stellav5,
    gerar_embedding_para_rag,
    gerar_card_material,
    obter_pontuacao_maxima,
    gerar_payload_para_frontend
)

from src.recommendation import get_recommendations

# --- CABEÇALHO ---
exibir_cabecalho()
st.markdown("### Devolutiva com Base Padronizada (Revisada)")

# --- CARREGAMENTO DE DADOS ---
modelo_st = carregar_modelo_st()
df_revisto = carregar_devolutivas_revisto()
df_rubricas = carregar_rubricas()
index = carregar_index("models/odas_index_stellav5.faiss")
df_odas = carregar_metadados("models/metadados_odas_stellav5.pkl")
st.success("Dados carregados com sucesso!")

# --- INTERFACE ---
# --- FILTROS DE PREFERÊNCIA ---
st.markdown("#### 2. Preferências de Aprendizagem")

opcoes_suporte = ["Videoaula", "Podcasts", "Artigos", "Livros"]

preferencias_positivas = st.multiselect(
    "Eu gosto de aprender com:",
    options=opcoes_suporte,
    help="Selecione os tipos de materiais que você prefere receber como recomendação."
)

preferencias_negativas = st.multiselect(
    "Eu NÃO gosto de aprender com:",
    options=opcoes_suporte,
    help="Os tipos selecionados aqui serão excluídos das recomendações finais."
)

st.markdown("#### 1. Preencha os dados para gerar a devolutiva")
dimensao = st.selectbox("Escolha a dimensão:", sorted(df_revisto["Dimensão"].dropna().unique()))
subdimensoes = sorted(df_revisto[df_revisto["Dimensão"] == dimensao]["Subdimensão"].dropna().unique())
subdimensao = st.selectbox("Escolha a subdimensão:", subdimensoes)
pontuacao_max = obter_pontuacao_maxima(df_rubricas, dimensao, subdimensao)
pontuacao = st.slider("Pontuação:", 0, pontuacao_max, min(17, pontuacao_max))

# --- BOTÃO ---
if st.button("Gerar devolutiva e recomendações"):
    texto = gerar_texto_padronizado_stellav5(df_revisto, df_rubricas, pontuacao, dimensao, subdimensao)

    if texto is None:
        st.warning("Texto padronizado não encontrado para a rubrica e faixa correspondente.")
    else:
        st.markdown(texto)

        # Geração de recomendações
        with st.spinner("Buscando materiais recomendados com base na devolutiva..."):
            texto_rico = texto
            artigos, videos, audios, visuais, interativos = get_recommendations(
                modelo_st, index, df_odas, df_rubricas, pontuacao, dimensao, subdimensao, texto_rico, "Modelo StellaV5"
            )

        # Exibição dos Resultados
        st.markdown("---")
        st.header("Materiais Recomendados")
        st.info("Resultados gerados pelo **Modelo StellaV5 (Base Padronizada Revisada)**.")

        # --- ORGANIZAÇÃO DAS CATEGORIAS COM BASE NA PREFERÊNCIA ---
        categorias = {
            "Artigos": ("📚", "Textos e Artigos", artigos),
            "Videoaula": ("🎥", "Vídeos e Aulas", videos),
            "Podcasts": ("🎧", "Áudios e Podcasts", audios),
            "Visuais": ("📊", "Materiais Visuais", visuais),
            "Interativos": ("🎮", "Materiais Interativos", interativos)
        }

        ordem_exibicao = preferencias_positivas + [k for k in categorias if k not in preferencias_positivas]

        for chave in preferencias_negativas:
            if chave in categorias:
                categorias[chave] = (categorias[chave][0], categorias[chave][1], pd.DataFrame())

        for chave in ordem_exibicao:
            emoji, titulo, df_categoria = categorias[chave]
            if not df_categoria.empty:
                st.markdown(f"#### {emoji} {titulo}")
                for i, row in enumerate(df_categoria.itertuples()):
                    st.markdown(gerar_card_material(row._asdict(), i))

        # --- GERAR PAYLOAD PARA FRONT-END (sem exibir no Streamlit) ---
        # Junta todos os materiais exibidos (já filtrados) em uma lista única
        materiais_final = pd.concat(
            [categorias[k][2] for k in ordem_exibicao if not categorias[k][2].empty],
            ignore_index=True
        )

        # Constrói a lista de recomendações no formato do app
        lista_recomendacoes = []
        for row in materiais_final.itertuples():
            dados = row._asdict()
            lista_recomendacoes.append({
                "title": dados.get("Título", ""),
                "description": dados.get("Resumo", ""),
                "type": dados.get("Suporte", "").lower(),
                "url": dados.get("Link fixo", ""),
                "workload": dados.get("Descricao_duracao", "")
            })

        # Gera o dicionário final
        payload_final = gerar_payload_para_frontend(texto, lista_recomendacoes)

        # Opcional: exibe no terminal para debug ou integração com API
        print(payload_final)
