# Home.py

import streamlit as st
from src.utils import exibir_cabecalho

# --- CONFIGURAÇÃO DA PÁGINA (DEVE SER O PRIMEIRO COMANDO) ---
st.set_page_config(
    page_title="RAG Gestores - Ambiente de testes",
    page_icon="📘",
    layout="wide"
)

# A barra lateral agora será gerada automaticamente pelo Streamlit
# com base nos arquivos da pasta 'pages'. Não precisamos mais configurá-la aqui.

# --- CONTEÚDO DA PÁGINA INICIAL ---
exibir_cabecalho()

st.header("RAG Gestores - Ambiente de testes")
st.markdown(
    """
    Ambiente de teste da RAG:

    1.  **Recomendação Individual:** Gere devolutivas personalizadas com base em uma pontuação e receba uma lista de materiais de formação recomendados.
    
    2.  **Devolutiva Geral:** Obtenha um texto de síntese consolidado para as dimensões "Planejamento Pedagógico" ou "Pessoal-Relacional".

    **Como usar:**
    - Utilize o menu de navegação na barra lateral à esquerda para acessar as diferentes ferramentas.
    """
)