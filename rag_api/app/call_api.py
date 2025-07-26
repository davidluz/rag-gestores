import requests, json

TOKEN = "sk-yP7gL0bXfT29vqHkJREcA1NzWuK4qDms"
URL   = "http://localhost:8000/recommend"

payload = {
    "pontuacao": 1,
    "dimensao": "Dimensão pedagógica",
    "subdimensao": "Planejamento pedagógico"
}

resp = requests.post(URL, json=payload,
                     headers={"Authorization": f"Bearer {TOKEN}"})
print("Status:", resp.status_code)
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
