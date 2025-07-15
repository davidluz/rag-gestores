# 📘 Plataforma de Apoio à Gestão Pedagógica

Este projeto fornece uma API web construída com FastAPI para auxiliar gestores pedagógicos na análise de desempenho e na formação continuada de professores. A ferramenta oferece devolutivas personalizadas e recomenda materiais de estudo com base em rubricas de avaliação.

## ✨ Funcionalidades Principais

* **Geração de Devolutivas Personalizadas:** Cria textos detalhados com pontos fortes, pontos a avançar e necessidades formativas com base em pontuações de avaliação.
* **Recomendação Inteligente de Materiais:** Utiliza busca vetorial (Sentence Transformers + FAISS) e um sistema de re-ranking ponderado para sugerir os materiais mais relevantes da base de dados.
* **Comparação de Modelos de IA:** Permite ao usuário alternar em tempo real entre diferentes motores de recomendação (Legado, Busca Simples e Avançado) para comparar a qualidade e a filosofia de cada abordagem.
* **Síntese de Devolutivas Gerais:** Usa um modelo de linguagem generativo (OpenAI GPT) para criar um texto consolidado para equipes.
* **Rastreamento de Experimentos:** Integração com MLflow para versionar, analisar e comparar cada ciclo de treinamento dos modelos de recomendação.

## 📂 Estrutura do Projeto

O projeto é organizado com uma estrutura modular para separar a preparação dos dados da aplicação final:

-   `PROJETO_MORI/` (Pasta Raiz)
    -   `data_source/`: Contém os dados brutos e originais.
    -   `notebooks/`: Para exploração de dados e desenvolvimento interativo (Jupyter/Colab).
    -   `scripts/`: Contém os pipelines automatizáveis e finais (ex: `train_model.py`).
    -   `streamlit_app/`: Código reutilizável e modelos pré-treinados.
        -   `models/`: Modelos (`.faiss`, `.pkl`) utilizados pelas funções de recomendação.
        -   `src/`: Módulos de lógica (`utils.py`, `recommendation.py`).
    -   `mlruns/`: Pasta criada pelo MLflow para armazenar os resultados dos experimentos (ignorada pelo Git).
    -   `README.md`, `requirements.txt`, `.gitignore`

## 🛠️ Tecnologias Utilizadas

* **API:** FastAPI
* **Manipulação de Dados:** Pandas, NumPy
* **IA & Busca Semântica:** Sentence-Transformers, FAISS, PyTorch
* **MLOps:** MLflow
* **Síntese de Texto:** OpenAI API

## 🚀 Guia de Uso e Execução

O projeto tem dois fluxos de trabalho principais: **1. Treinamento/Geração de Modelos** e **2. Execução da Aplicação Web**.

### 1. Setup do Ambiente (Feito apenas uma vez)

```bash
# Clone o repositório e entre na pasta raiz
git clone https://github.com/davidluz/rag-gestores.git
cd rag-gestores

# (Opcional, mas recomendado) Ative o Git LFS para arquivos grandes
git lfs install
git lfs pull

# Crie e ative um ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale todas as dependências
pip install -r requirements.txt
```

### 2. Executando a API

Após instalar as dependências, inicie o servidor FastAPI com:

```bash
uvicorn api.main:app
```
