import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Gerar dados de sequência
def generate_sequence(start, end):
    return np.array([i for i in range(start, end+1, 3)])

# Preparar os dados para treinamento/teste
def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        # encontrar o fim do padrão
        end_ix = i + n_steps
        # checar se estamos além da sequência
        if end_ix > len(sequence)-1:
            break
        # coletar input e output partes da sequência
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Definir parâmetros
n_steps = 3
sequence = generate_sequence(0, 100)
X, y = prepare_data(sequence, n_steps)

# Redimensionar X para o formato [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Definir modelo
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Treinar modelo
model.fit(X, y, epochs=200, verbose=0)

# Demonstração de previsão
x_input = np.array([81, 84, 87])
x_input = x_input.reshape((1, n_steps, 1))
yhat = model.predict(x_input, verbose=0)
print(f'Next number prediction after [81, 84, 87] is {yhat[0][0]:.0f}')
