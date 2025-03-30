import os

def selecionar_devolutiva(score: float) -> str:
    # Defina os intervalos e os nomes dos arquivos
    if score < 3:
        arquivo = "1_Devolutiva_dim_pedag_planejamento.md"
    elif 3 <= score < 4:
        arquivo = "2_Devolutiva_dim_pedag_implementacao.md"
    else:
        arquivo = "3_Devolutiva_dim_pedag_monitoriamento"

    # Construa o caminho completo (ajuste conforme sua estrutura)
    caminho = os.path.join("data", "markdowns", arquivo)
    
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            conteudo = f.read()
        return conteudo
    except FileNotFoundError:
        return "Arquivo de devolutiva não encontrado."
