import requests

body = {
    "open": 17.25,
    "high": 18.00,
    "low": 16.50,
    "volume": 708486000
}

response = requests.post(url='http://127.0.0.1:8000/predict', json=body)
print(response.json())  # Espera un JSON con 'predicted_close'
