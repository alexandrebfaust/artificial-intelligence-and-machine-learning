import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import time

# Carregar dados do arquivo CSV
dados = pd.read_csv('src/TSLA.csv')
dados['Date'] = pd.to_datetime(dados['Date'])
dados['Date'] = (dados['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')  # Unix timestamp

#print(dados)

precos_abertura = dados['Open'].values

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
precos_norm = scaler.fit_transform(precos_abertura.reshape(-1, 1))

# Preparando os dados para a LSTM: usando janelas de tempo
window_size = 50
X = []
y = []
for i in range(len(precos_norm) - window_size):
    X.append(precos_norm[i:i + window_size, 0])
    y.append(precos_norm[i + window_size, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Criando o modelo LSTM
model = Sequential()
model.add(LSTM(50, activation='sigmoid', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Treinando o modelo
start_time = time.time()
model.fit(X, y, epochs=100, verbose=0)
training_time = time.time() - start_time
print(f"Tempo de treinamento: {training_time:.2f} segundos.")

# Prever o valor de abertura para o próximo dia
last_window = X[-1:]
next_day_pred_norm = model.predict(last_window)
next_day_pred = scaler.inverse_transform(next_day_pred_norm)

print(f'Previsão do preço de abertura para o próximo dia: {next_day_pred[0][0]}')

# Preço médio previsto para abertura:
# Sigmoid: 954,281921386718
# ReLU: 982,167175292968

#Date,Open
#2022-03-22,930.000000
#2022-03-23,979.940002
#2022-03-24,1009.729980