from typing import Dict, Any, Tuple, List

class RAGService:
    def __init__(self) -> None:
        self.model = None     # placeholder

    def infer(self, features: Dict[str, Any]) -> Tuple[str, List[str]]:
        texto = "Devolutiva simulada"
        materiais = ["Artigo A", "Vídeo B"]
        return texto, materiais

#def infer(self, features: Dict[str, Any]) -> Tuple[str, List[str]]:
    # 1. Pré-processamento dos inputs
#    pontuacao = features["pontuacao"]
#    # 2. Chamada ao modelo
#    texto = self.model.generate_feedback(features)
#    materiais = self.model.recommend_materials(features)
#    # 3. Pós-processamento (opcional)
#    return texto, materiais