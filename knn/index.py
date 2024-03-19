from flask import Flask, request, jsonify, send_from_directory
from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

# Conectar ao MongoDB
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
client = MongoClient(DB_CONNECTION_STRING)
db = client['academia']
alunos = db.alunos

# Testa a conexão
total_documents = alunos.count_documents({})
print("Total de documentos na coleção alunos:", total_documents)

def categoria(predicao):
    return 'Ganho de Massa' if predicao == 0 else 'Perda de Peso'

def treinar_knn():
    # Busca todos os alunos no banco de dados
    dados_alunos = list(alunos.find())
    # Prepara os dados para o treinamento
    X = np.array([[a['altura'], a['peso'], a['IMC'], a['gordura_corporal']] for a in dados_alunos if 'categoria' in a])
    y = np.array([a['categoria'] for a in dados_alunos if 'categoria' in a])
    # Cria e treina o KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

@app.route('/aluno', methods=['POST'])
def add_aluno():
    if request.method == 'POST':
        dados_aluno = request.json

        print(dados_aluno)

        # Verifica campos obrigatórios
        campos_obrigatorios = ['nome', 'usuario', 'peso', 'altura', 'gordura_corporal']
        if not all(campo in dados_aluno for campo in campos_obrigatorios):
            print("Faltam campos obrigatórios", dados_aluno, campos_obrigatorios)
            return jsonify({'erro': 'Faltam campos obrigatórios'}), 400

        # Verifica se o usuário já existe
        usuario_existente = alunos.find_one({'usuario': dados_aluno.get('usuario')})
        if usuario_existente:
            return jsonify({'erro': 'Usuário já existe'}), 400

        # Calcula o IMC
        altura_m = dados_aluno['altura'] / 100
        dados_aluno['IMC'] = round(dados_aluno['peso'] / (altura_m ** 2), 2)

        # Insere um novo aluno
        resultado = alunos.insert_one({k: v for k, v in dados_aluno.items()})

        # Treina o KNN com os dados atualizados
        knn = treinar_knn()

        # Realiza a predição para o aluno inserido
        aluno_data = np.array([[dados_aluno['altura'], dados_aluno['peso'], dados_aluno['IMC'], dados_aluno['gordura_corporal']]])
        print("Aluno data", aluno_data)
        predicao = knn.predict(aluno_data)
        print(predicao)
        print("Aluno", resultado.inserted_id, "Predição:", int(predicao[0]))

        # Atualiza o aluno com a categoria predita
        alunos.update_one({'_id': resultado.inserted_id}, {'$set': {'categoria': int(predicao[0])}})

        return jsonify({
            'id': str(resultado.inserted_id), 
            'categoria': int(predicao[0]), 
            'categoria_titulo': categoria(int(predicao[0]))
            }), 201

if __name__ == '__main__':
    app.run(debug=False)
