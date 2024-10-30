from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Usar regresor en lugar de clasificador
from joblib import dump
import pandas as pd
import pathlib

# Cargar el nuevo conjunto de datos
df = pd.read_csv(pathlib.Path('data/visa_stocks.csv'))  # Cambia el nombre del archivo CSV
df['Date'] = pd.to_datetime(df['Date'])  # Convertir la columna de fecha a datetime
df.set_index('Date', inplace=True)  # Usar la fecha como índice

# Seleccionar las características (features) y la variable objetivo (target)
X = df[['Open', 'High', 'Low', 'Volume']]  # Selecciona las características que usarás
y = df['Close']  # Establece el precio de cierre como variable objetivo

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training model.. ')
# Cambiar a un RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0)
regressor.fit(X_train, y_train)
print('Saving model..')

# Guardar el modelo entrenado
dump(regressor, pathlib.Path('model/visa_stocks-v1.joblib'))  # Cambia el nombre del modelo
