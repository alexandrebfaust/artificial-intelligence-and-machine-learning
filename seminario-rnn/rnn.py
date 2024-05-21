import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import time  # Importando a biblioteca time

# Carregar dados de um arquivo CSV
dados = pd.read_csv('dados.csv')
dias = dados['dia'].values
valores = dados['valor'].values

# Normalizando os dados para o treinamento da rede
max_valor = np.max(valores)
valores_norm = valores / max_valor

# Preparando os dados para a LSTM: [samples, time steps, features]
dias = dias.reshape((len(dias), 1, 1))
valores_norm = valores_norm.reshape((len(valores_norm), 1))

# Criando o modelo LSTM
model = Sequential()
model.add(Input(shape=(1, 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Inicia a contagem de tempo
start_time = time.time()

# Treinando o modelo
model.fit(dias, valores_norm, epochs=1000, verbose=0)

# Finaliza a contagem de tempo após o treinamento
training_time = time.time() - start_time
print(f"Tempo de treinamento: {training_time:.2f} segundos.")

# Inicia a contagem de tempo para previsão
start_time = time.time()

# Prever o valor no dia 5
dia_5 = np.array([5]).reshape((1, 1, 1))
previsao_norm = model.predict(dia_5)

# Finaliza a contagem de tempo para previsão
prediction_time = time.time() - start_time
print(f"Tempo de previsão: {prediction_time:.4f} segundos.")

previsao = previsao_norm * max_valor
print(f'Previsão do valor da ação no dia 5: {previsao[0][0]}')


#ReLU: 1.303592324256897
#Sigmoid: 1.0913374423980713

