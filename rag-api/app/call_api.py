# call_api.py
import requests

URL = "http://127.0.0.1:8000/gerar"
payload = {
    "pontuacao": 8.0,
    "subdimensao": "Motivação",
    "preferencias_do_usuario": ["pdf"],
    "dimensao": "Empatia"
}

resp = requests.post(URL, json=payload, timeout=5)
print("Status:", resp.status_code)
print("Resposta:", resp.json())
