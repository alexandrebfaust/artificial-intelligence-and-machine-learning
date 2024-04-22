import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('Iris.csv')
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Testar pelo menos duas configurações de rede
configurations = [
    (1, 5),  # 1 camada escondida com 5 neurônios
    (2, [10, 5])  # 2 camadas escondidas com 10 e 5 neurônios respectivamente
]

# Utilizar 10 fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

best_config = None
best_score = 0

for i, config in enumerate(configurations):
    layers = tuple(config[1]) if isinstance(config[1], list) else (config[1],)
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=10000, random_state=42)
    scores = cross_val_score(mlp, X_scaled, y_encoded, cv=kf)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_config = layers
    print(f"Configuração {i+1}: Camadas {config[1]} - Acurácia média: {scores.mean():.4f}")

mlp_best = MLPClassifier(hidden_layer_sizes=best_config, max_iter=10000, random_state=42)
mlp_best.fit(X_scaled, y_encoded)

def predict_new_data():
    new_data = list(map(float, input("\nDigite as características da flor Iris para predição (SepalLength, SepalWidth, PetalLength, PetalWidth): ").split()))
    new_data_df = pd.DataFrame([new_data], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    new_data_scaled = scaler.transform(new_data_df)
    prediction = mlp_best.predict(new_data_scaled)
    predicted_species = label_encoder.inverse_transform(prediction)
    print(f"A espécie prevista é: {predicted_species[0]}")

while True:
    predict_new_data()

#Dados para teste
# 5.1 3.5 1.4 0.2 -> Iris-setosa
# 6.7 3.0 5.2 2.3 -> Iris-virginica