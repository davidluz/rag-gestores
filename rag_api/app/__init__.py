"""Permite importações relativas curtas (e força carregamento rápido em produção)."""
from importlib import import_module

# Opcional: garante que o rag_service seja pré‑carregado quando o pacote é importado.
import_module("rag_api.app.rag_service")
