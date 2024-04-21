from pymongo import MongoClient
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import io
from dotenv import load_dotenv
import os

# Conexão com o MongoDB
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
client = MongoClient(DB_CONNECTION_STRING)
db = client['academia']
collection = db['alunos']

alunos = list(collection.find({}))

data = {
    'Nome': [aluno['usuario'] for aluno in alunos],
    'Longitude': [aluno['IMC'] for aluno in alunos],
    'Latitude': [aluno['gordura_corporal'] for aluno in alunos],
    'Categoria': [aluno['categoria'] for aluno in alunos]
}
places = pd.DataFrame(data)

coords = places[['Longitude', 'Latitude']].values

categoriaFlag = np.append(places['Categoria'].values, [2, 2, 2, 2], axis=0)

colours = ['#FF5733' if categoria == 0 else '#33CFFF' for categoria in categoriaFlag[:-4]] + ['w', 'w', 'w', 'w']

fig = plt.figure(figsize=(8, 6.8), dpi=100)  

vor = Voronoi(coords)
voronoi_plot_2d(vor, show_vertices=False, point_size=2, figure=fig)
fig = plt.figure(figsize=(8, 6.8), dpi=100)
for j in range(len(coords)):
    region = vor.regions[vor.point_region[j]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), colours[j])

plt.plot(coords[:,0], coords[:,1], 'ko')
plt.xlim(min(coords[:,0])-1, max(coords[:,0])+1), plt.ylim(min(coords[:,1])-1, max(coords[:,1])+1)

for i in range(len(places)):
    plt.annotate(places['Nome'][i], (coords[i,0], coords[i,1]), xytext=(coords[i,0]-0.5, coords[i,1]+0.5))

img_bytes = io.BytesIO()
plt.savefig(img_bytes, format='png', dpi=100)
img_bytes.seek(0)
plt.close(fig)