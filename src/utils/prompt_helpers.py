import requests

def chamar_claude(prompt: str) -> str:
    # URL do API Gateway que encaminha para o modelo Claude
    api_gateway_url = "https://seu-api-gateway.amazonaws.com/prod/claude"
    headers = {
        "Content-Type": "application/json",
        # Adicione quaisquer cabeçalhos de autenticação que forem necessários
    }
    payload = {"prompt": prompt}
    response = requests.post(api_gateway_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        raise Exception(f"Erro na chamada do Claude: {response.status_code} - {response.text}")
