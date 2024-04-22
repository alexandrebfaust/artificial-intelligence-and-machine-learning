import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Carregar o dataset Iris
df = pd.read_csv('Iris.csv')
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df['Species'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Testar pelo menos duas alternativas de número de vizinhos mais próximos
k_values = [3, 5]
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X, y_encoded, cv=kf)
    results[k] = np.mean(cv_scores)
    print(f'Média da acurácia para k={k}: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})')

best_k = max(results, key=results.get)
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X, y_encoded)
print(f"Modelo treinado com k={best_k}")

def predict_new_data():
    new_data = list(map(float, input("\nDigite as características da flor Iris para predição (SepalLength, SepalWidth, PetalLength, PetalWidth): ").split()))
    new_data = np.array(new_data).reshape(1, -1)
    prediction = model.predict(new_data)
    predicted_species = label_encoder.inverse_transform(prediction)
    print(f"A espécie prevista é: {predicted_species[0]}")

while True:
    predict_new_data()

#Dados para teste
# 5.1 3.5 1.4 0.2 -> Iris-setosa
# 6.7 3.0 5.2 2.3 -> Iris-virginica