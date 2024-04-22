import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np

data = pd.read_csv('Iris.csv')
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

kernels = ['linear', 'rbf']
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}

for kernel in kernels:
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel))
    scores = cross_val_score(model, X, y_encoded, cv=kf)
    results[kernel] = scores.mean()
    print(f"Acurácia para kernel {kernel}: {scores.mean():.2f} ± {scores.std():.2f}")

best_kernel = max(results, key=results.get)
final_model = make_pipeline(StandardScaler(), SVC(kernel=best_kernel))
final_model.fit(X, y_encoded)

def predict_new_data():
    new_data_input = input("\nDigite as características da flor Iris para predição (SepalLength, SepalWidth, PetalLength, PetalWidth): ")
    new_data_list = list(map(float, new_data_input.split()))
    new_data = pd.DataFrame([new_data_list], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    prediction = final_model.predict(new_data)
    predicted_species = label_encoder.inverse_transform(prediction)
    print(f"A espécie prevista é: {predicted_species[0]}")

while True:
    predict_new_data()

#Dados para teste
# 5.1 3.5 1.4 0.2 -> Iris-setosa
# 6.7 3.0 5.2 2.3 -> Iris-virginica