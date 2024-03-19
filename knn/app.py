import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from dotenv import load_dotenv
import os
import json
import io

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
    X = np.array([[a['IMC'], a['gordura_corporal']] for a in dados_alunos if 'categoria' in a])
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
        aluno_data = np.array([[dados_aluno['IMC'], dados_aluno['gordura_corporal']]])
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

@app.route('/aluno/<id>', methods=['GET'])
def get_aluno(id):
    if request.method == 'GET':
        aluno = alunos.find_one({'_id': ObjectId(id)})
        if aluno:
            aluno['_id'] = str(aluno['_id'])
            aluno['categoria_titulo'] = categoria(aluno['categoria'])
            return jsonify(aluno), 200
        else:
            return jsonify({'erro': 'Aluno não encontrado'}), 404

@app.route('/stats') # Cria um gráfico de barras com a distribuição das categorias
def estatisticas():
    categorias = [aluno['categoria'] for aluno in alunos.find()]
    categorias_count = [categorias.count(0), categorias.count(1)]

    labels = ['Ganho de Massa', 'Perda de Peso']
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.bar(labels, categorias_count, color=['blue', 'orange'])
    axis.set_title('Distribuição de Alunos por Categoria de Treino')
    axis.set_xlabel('Categoria de Treino')
    axis.set_ylabel('Número de Alunos')
    axis.set_xticks(labels)
    axis.set_yticks(np.arange(0, max(categorias_count) + 1, step=1))

    output = io.BytesIO()
    Figure.savefig(fig, output, format='png')
    output.seek(0)

    return send_file(output, mimetype='image/png')

@app.route('/voronoi')
def voronoi_diagram():
    # Busca dados dos alunos para usar como pontos no diagrama de Voronoi
    pontos = np.array([[aluno['IMC'], aluno['gordura_corporal']] for aluno in alunos.find()])

    if len(pontos) < 2:
        return jsonify({'erro': 'Não há dados suficientes para gerar o diagrama de Voronoi.'}), 400

    # Gera o diagrama de Voronoi
    vor = Voronoi(pontos)

    # Cria uma figura e desenha o diagrama de Voronoi
    fig = plt.figure()
    voronoi_plot_2d(vor, show_vertices=False, point_size=3)
    plt.title('Diagrama de Voronoi : IMC x Gordura Corporal')

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig)

    return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False)
