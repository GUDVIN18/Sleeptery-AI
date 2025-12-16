import requests

url = "https://api.edenai.run/v2/text/embeddings/"

payload = {
    "response_as_dict": True,
    "attributes_as_list": False,
    "show_base_64": False,
    "show_original_response": False,

    "providers": ["cohere"],

    "cohere": {
        "model": "embed-multilingual-v3.0"
    },

    "texts": [
        "Это тест для русского embedding. Проверяем модель v3.0."
    ]
}

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmM4YjE0OWItNTQ3OS00MDE5LWE2MjAtODZjYzRjZGVjNTRlIiwidHlwZSI6ImFwaV90b2tlbiJ9.i2SjhGsmklcGX87YuIqOOBQIVG_Bn2TG3DpomClXPYA"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())