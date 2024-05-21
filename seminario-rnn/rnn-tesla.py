import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

# Carregar dados do arquivo CSV
dados = pd.read_csv('src/TSLA.csv')
dados['Date'] = pd.to_datetime(dados['Date'])
precos_abertura = dados['Open'].values

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
precos_norm = scaler.fit_transform(precos_abertura.reshape(-1, 1))

# Preparando os dados para a LSTM: usando janelas de tempo
window_size = 5
X = []
y = []
dates = []
for i in range(len(precos_norm) - window_size):
    X.append(precos_norm[i:i + window_size, 0])
    y.append(precos_norm[i + window_size, 0])
    dates.append(dados['Date'].iloc[i + window_size])  # Salvando a data correspondente

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.10, random_state=42)

# Criando o modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Treinando o modelo
start_time = time.time()
model.fit(X_train, y_train, epochs=100, verbose=0)
training_time = time.time() - start_time
print(f"Tempo de treinamento: {training_time:.2f} segundos.")

# Avaliando o modelo no conjunto de treinamento e teste
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

print(f'MSE de treinamento: {train_mse}')
print(f'MSE de teste: {test_mse}')

# Comparando previsões de teste com valores reais
test_pred_values = scaler.inverse_transform(test_pred)
y_test_values = scaler.inverse_transform(y_test.reshape(-1, 1))

print('Datas e Previsões de teste (Data, Real vs. Previsto):')
for date, real, pred in zip(dates_test, y_test_values, test_pred_values):
    print(f'Data: {date.strftime("%Y-%m-%d")}, Real: {real[0]:.2f}, Previsto: {pred[0]:.2f}')
