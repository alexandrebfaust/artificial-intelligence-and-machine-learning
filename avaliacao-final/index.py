import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, make_scorer
import joblib
import os

plot_graphs = True

# Lista de caminhos dos arquivos
file_list = ['src/08.csv', 'src/09.csv', 'src/28.csv']

def run_ml_pipeline(file_path):

    # Carregar o conjunto de dados fornecido
    data = pd.read_csv(file_path)
    filename = file_path.replace('src/', '').replace('.csv', '')

    # Exibir informações básicas sobre o conjunto de dados
    data_info = data.info()
    data_head = data.head()

    # Calcular a matriz de correlação
    correlation_matrix = data.corr()

    # Plotar o mapa de calor
    if plot_graphs:
        plt.figure(figsize=(15, 15))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f')
        plt.title('Heatmap da Matriz de Correlação')
        plt.show()

    # Identificar características altamente correlacionadas (limite > 0.85)
    correlation_threshold = 0.85
    high_corr_pairs = []

    # Iterar sobre a matriz de correlação
    for i in range(correlation_matrix.shape[0]):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

    # Criar um conjunto para armazenar as colunas a serem removidas
    columns_to_remove = set()

    # Decidir quais colunas remover (mantendo a primeira coluna e removendo a segunda em cada par)
    for col1, col2 in high_corr_pairs:
        columns_to_remove.add(col2)

    # Remover as colunas identificadas do conjunto de dados
    data_reduced = data.drop(columns=columns_to_remove)

    # Separar características e variável alvo
    X = data_reduced.drop(columns=['0'])
    y = data_reduced['0']

    # Converter y de {1, 2} para {0, 1}
    y = y - 1

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Padronizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar um modelo de Random Forest para obter importâncias das características
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Obter importâncias das características
    feature_importances = rf_model.feature_importances_

    # Criar um DataFrame para importâncias das características
    feature_importance_df = pd.DataFrame({
        'Característica': X.columns,
        'Importância': feature_importances
    }).sort_values(by='Importância', ascending=False)

    # Exibir o DataFrame de importâncias das características
    print("Importâncias das Características:\n", feature_importance_df.head(10))

    # Plotar a distribuição das 10 características mais importantes
    top_features = feature_importance_df.head(10)['Característica'].tolist()

    # Plotagem
    if plot_graphs:
        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes = axes.flatten()

        for i, feature in enumerate(top_features):
            sns.histplot(data[feature], bins=30, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribuição da Característica {feature}')
            
        plt.tight_layout()
        plt.show()

    # Selecionar as principais características importantes
    selected_features = feature_importance_df.head(10)['Característica'].tolist()
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Padronizar as características selecionadas
    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)

    # Definir grades de parâmetros para GridSearchCV
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    mlp_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    print("Inicializando modelos...")

    # Inicializar modelos
    knn = KNeighborsClassifier()
    mlp = MLPClassifier(max_iter=10000, random_state=42)
    svm = SVC(probability=True, random_state=42)

    print("Verificando arquivos existentes do GridSearchCV...")

    # Verificar se arquivos do GridSearchCV existem
    knn_grid_search_file = 'fit/'+filename+'_knn_grid_search.pkl'
    mlp_grid_search_file = 'fit/'+filename+'_mlp_grid_search.pkl'
    svm_grid_search_file = 'fit/'+filename+'_svm_grid_search.pkl'

    if os.path.exists(knn_grid_search_file):
        knn_grid_search = joblib.load(knn_grid_search_file)
        print("Modelo KNN GridSearchCV existente carregado.")
    else:
        knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='roc_auc')
        print("Ajustando KNN GridSearchCV...")
        knn_grid_search.fit(X_train_selected_scaled, y_train)
        joblib.dump(knn_grid_search, knn_grid_search_file)

    if os.path.exists(mlp_grid_search_file):
        mlp_grid_search = joblib.load(mlp_grid_search_file)
        print("Modelo MLP GridSearchCV existente carregado.")
    else:
        mlp_grid_search = GridSearchCV(mlp, mlp_param_grid, cv=5, scoring='roc_auc')
        print("Ajustando MLP GridSearchCV...")
        mlp_grid_search.fit(X_train_selected_scaled, y_train)
        joblib.dump(mlp_grid_search, mlp_grid_search_file)

    if os.path.exists(svm_grid_search_file):
        svm_grid_search = joblib.load(svm_grid_search_file)
        print("Modelo SVM GridSearchCV existente carregado.")
    else:
        svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='roc_auc')
        print("Ajustando SVM GridSearchCV...")
        svm_grid_search.fit(X_train_selected_scaled, y_train)
        joblib.dump(svm_grid_search, svm_grid_search_file)

    # Obter os melhores parâmetros e pontuações
    knn_best_params = knn_grid_search.best_params_
    mlp_best_params = mlp_grid_search.best_params_
    svm_best_params = svm_grid_search.best_params_

    knn_best_score = knn_grid_search.best_score_
    mlp_best_score = mlp_grid_search.best_score_
    svm_best_score = svm_grid_search.best_score_

    print(f"Melhores Parâmetros do KNN: {knn_best_params}, Melhor AUC-ROC: {knn_best_score}")
    print(f"Melhores Parâmetros do MLP: {mlp_best_params}, Melhor AUC-ROC: {mlp_best_score}")
    print(f"Melhores Parâmetros do SVM: {svm_best_params}, Melhor AUC-ROC: {svm_best_score}")

    # Definir uma função para avaliar modelos usando validação cruzada
    def evaluate_model(model, X, y):
        accuracy = cross_val_score(model, X, y, cv=10, scoring='accuracy').mean()
        precision = cross_val_score(model, X, y, cv=10, scoring=make_scorer(precision_score, zero_division=0, average='macro')).mean()
        recall = cross_val_score(model, X, y, cv=10, scoring=make_scorer(recall_score, zero_division=0, average='macro')).mean()
        f1 = cross_val_score(model, X, y, cv=10, scoring=make_scorer(f1_score, zero_division=0, average='macro')).mean()
        auc = cross_val_score(model, X, y, cv=10, scoring='roc_auc_ovr').mean()
        return accuracy, precision, recall, f1, auc

    # Avaliar cada classificador
    knn_results = evaluate_model(knn, X_train_selected_scaled, y_train)
    mlp_results = evaluate_model(mlp, X_train_selected_scaled, y_train)
    svm_results = evaluate_model(svm, X_train_selected_scaled, y_train)

    # Compilar os resultados em um DataFrame
    results_df = pd.DataFrame({
        'Classificador': ['KNN', 'MLP', 'SVM'],
        'Acurácia': [knn_results[0], mlp_results[0], svm_results[0]],
        'Precisão': [knn_results[1], mlp_results[1], svm_results[1]],
        'Recall': [knn_results[2], mlp_results[2], svm_results[2]],
        'F1-Score': [knn_results[3], mlp_results[3], svm_results[3]],
        'AUC-ROC': [knn_results[4], mlp_results[4], svm_results[4]]
    })

    print("Comparação de Desempenho dos Classificadores:\n", results_df)

    # Treinar o Classificador de Maioria
    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(X_train_scaled, y_train)

    # Prever no conjunto de teste
    y_dummy_pred = dummy_clf.predict(X_test_scaled)
    y_dummy_prob = dummy_clf.predict_proba(X_test_scaled)[:, 1]

    # Calcular métricas de baseline
    baseline_accuracy = accuracy_score(y_test, y_dummy_pred)
    baseline_precision = precision_score(y_test, y_dummy_pred, average='macro', zero_division=0)
    baseline_recall = recall_score(y_test, y_dummy_pred, average='macro', zero_division=0)
    baseline_f1 = f1_score(y_test, y_dummy_pred, average='macro', zero_division=0)
    baseline_auc = roc_auc_score(y_test, y_dummy_prob)

    baseline_metrics = {
        'Acurácia': baseline_accuracy,
        'Precisão': baseline_precision,
        'Recall': baseline_recall,
        'F1-Score': baseline_f1,
        'AUC-ROC': baseline_auc
    }

    print("Métricas de Baseline:\n", baseline_metrics)

    # Pontuações de baseline
    baseline_scores = [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1, baseline_auc]

    # Métricas
    metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    knn_scores = [knn_results[0], knn_results[1], knn_results[2], knn_results[3], knn_results[4]]
    mlp_scores = [mlp_results[0], mlp_results[1], mlp_results[2], mlp_results[3], mlp_results[4]]
    svm_scores = [svm_results[0], svm_results[1], svm_results[2], svm_results[3], svm_results[4]]

    x = range(len(metrics))

    if plot_graphs:
        plt.figure(figsize=(12, 6))
        plt.bar(x, baseline_scores, width=0.2, label='Baseline', align='center')
        plt.bar([p + 0.2 for p in x], knn_scores, width=0.2, label='KNN', align='center')
        plt.bar([p + 0.4 for p in x], mlp_scores, width=0.2, label='MLP', align='center')
        plt.bar([p + 0.6 for p in x], svm_scores, width=0.2, label='SVM', align='center')
        plt.xticks([p + 0.3 for p in x], metrics)
        plt.xlabel('Métricas')
        plt.ylabel('Pontuações')
        plt.title('Comparação de Desempenho dos Classificadores com Baseline')
        plt.legend(loc='upper left')
        plt.show()

    if plot_graphs:
        # Plotar curvas ROC
        plt.figure(figsize=(10, 6))

        # KNN
        knn_best_model = knn_grid_search.best_estimator_
        y_scores_knn = knn_best_model.predict_proba(X_test_selected_scaled)[:, 1]
        fpr_knn, tpr_knn, _ = roc_curve(y_test, y_scores_knn)
        roc_auc_knn = auc(fpr_knn, tpr_knn)
        plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')

        # MLP
        mlp_best_model = mlp_grid_search.best_estimator_
        y_scores_mlp = mlp_best_model.predict_proba(X_test_selected_scaled)[:, 1]
        fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_scores_mlp)
        roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
        plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC = {roc_auc_mlp:.2f})')

        # SVM
        svm_best_model = svm_grid_search.best_estimator_
        y_scores_svm = svm_best_model.predict_proba(X_test_selected_scaled)[:, 1]
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_scores_svm)
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')

        # Dummy
        fpr_dummy, tpr_dummy, _ = roc_curve(y_test, y_dummy_prob)
        roc_auc_dummy = auc(fpr_dummy, tpr_dummy)
        plt.plot(fpr_dummy, tpr_dummy, label=f'Baseline (AUC = {roc_auc_dummy:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curvas ROC (Receiver Operating Characteristic)')
        plt.legend(loc="lower right")
        plt.show()

# Executar o pipeline de ML para cada arquivo
for file_path in file_list:
    print(f"Executando pipeline de ML para o arquivo: {file_path}")
    run_ml_pipeline(file_path)
