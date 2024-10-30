from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Stock Price Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

# Cargar el modelo
model = load(pathlib.Path('model/visa_stocks-v1.joblib'))

# Definir el modelo de entrada
class InputData(BaseModel):
    open: float
    high: float
    low: float
    volume: int

# Definir el modelo de salida
class OutputData(BaseModel):
    predicted_close: float

@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    model_input = np.array([data.open, data.high, data.low, data.volume]).reshape(1, -1)
    predicted_price = model.predict(model_input)

    return {'predicted_close': predicted_price[0]}  # Devuelve el precio de cierre predicho
