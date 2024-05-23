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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

plot_graphs = False

# Load the provided dataset
file_path = 'src/08.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
if plot_graphs:
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title('Heatmap da Matriz de Correlação')
    plt.show()

# Identify highly correlated features (threshold > 0.85)
correlation_threshold = 0.85
high_corr_pairs = []

# Iterate over the correlation matrix
for i in range(correlation_matrix.shape[0]):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

# Create a set to hold columns to remove
columns_to_remove = set()

# Decide which columns to remove (keeping the first column and removing the second in each pair)
for col1, col2 in high_corr_pairs:
    columns_to_remove.add(col2)

# Remove the identified columns from the dataset
data_reduced = data.drop(columns=columns_to_remove)

# Separate features and target variable
X = data.drop(columns=['0'])
y = data['0']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model to get feature importances
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importance DataFrame
print("Feature Importances:\n", feature_importance_df.head(10))

# Plot distribution of the top 10 important features
top_features = feature_importance_df.head(10)['Feature'].tolist()

# Plotting
if plot_graphs:
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        sns.histplot(data[feature], bins=30, ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of Feature {feature}')
        
    plt.tight_layout()
    plt.show()

# Select top important features
selected_features = feature_importance_df.head(10)['Feature'].tolist()
X_selected = data[selected_features]

# Standardize the selected features
X_selected_scaled = scaler.fit_transform(X_selected)

# Define parameter grids for GridSearchCV
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

print("Initialize models...")

# Initialize models
knn = KNeighborsClassifier()
mlp = MLPClassifier(max_iter=10000, random_state=42)
svm = SVC(probability=True, random_state=42)


print("Initialize GridSearchCV...")

# Initialize GridSearchCV
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='roc_auc')
mlp_grid_search = GridSearchCV(mlp, mlp_param_grid, cv=5, scoring='roc_auc')
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='roc_auc')

# Fit the grid search to the data
print("Fitting GridSearchCV...")
print("KNN...")
knn_grid_search.fit(X_selected_scaled, y)
print("MLP...")
mlp_grid_search.fit(X_selected_scaled, y)
print("SVM...")
svm_grid_search.fit(X_selected_scaled, y)

# Get the best parameters and scores
knn_best_params = knn_grid_search.best_params_
mlp_best_params = mlp_grid_search.best_params_
svm_best_params = svm_grid_search.best_params_

knn_best_score = knn_grid_search.best_score_
mlp_best_score = mlp_grid_search.best_score_
svm_best_score = svm_grid_search.best_score_

print(f"KNN Best Params: {knn_best_params}, Best AUC-ROC: {knn_best_score}")
print(f"MLP Best Params: {mlp_best_params}, Best AUC-ROC: {mlp_best_score}")
print(f"SVM Best Params: {svm_best_params}, Best AUC-ROC: {svm_best_score}")

# 08.csv
# KNN Best Params: {'n_neighbors': 11, 'p': 2, 'weights': 'distance'}, Best AUC-ROC: 0.96984126984127
# MLP Best Params: {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'}, Best AUC-ROC: 0.9745714285714285
# SVM Best Params: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}, Best AUC-ROC: 0.9821587301587302

# Define a function to evaluate models using cross-validation
def evaluate_model(model, X, y):
    accuracy = cross_val_score(model, X, y, cv=10, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=10, scoring='precision_macro').mean()
    recall = cross_val_score(model, X, y, cv=10, scoring='recall_macro').mean()
    f1 = cross_val_score(model, X, y, cv=10, scoring='f1_macro').mean()
    auc = cross_val_score(model, X, y, cv=10, scoring='roc_auc_ovr').mean()
    return accuracy, precision, recall, f1, auc

# Evaluate each classifier
knn_results = evaluate_model(knn, X_selected_scaled, y)
mlp_results = evaluate_model(mlp, X_selected_scaled, y)
svm_results = evaluate_model(svm, X_selected_scaled, y)

# Compile the results into a DataFrame
results_df = pd.DataFrame({
    'Classifier': ['KNN', 'MLP', 'SVM'],
    'Accuracy': [knn_results[0], mlp_results[0], svm_results[0]],
    'Precision': [knn_results[1], mlp_results[1], svm_results[1]],
    'Recall': [knn_results[2], mlp_results[2], svm_results[2]],
    'F1-Score': [knn_results[3], mlp_results[3], svm_results[3]],
    'AUC-ROC': [knn_results[4], mlp_results[4], svm_results[4]]
})

print("Classifier Performance Comparison:\n", results_df)

# Train the Majority Class Classifier
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_dummy_pred = dummy_clf.predict(X_test_scaled)
y_dummy_prob = dummy_clf.predict_proba(X_test_scaled)[:, 1]

# Calculate baseline metrics
baseline_accuracy = accuracy_score(y_test, y_dummy_pred)
baseline_precision = precision_score(y_test, y_dummy_pred, average='macro')
baseline_recall = recall_score(y_test, y_dummy_pred, average='macro')
baseline_f1 = f1_score(y_test, y_dummy_pred, average='macro')
baseline_auc = roc_auc_score(y_test, y_dummy_prob)

baseline_metrics = {
    'Accuracy': baseline_accuracy,
    'Precision': baseline_precision,
    'Recall': baseline_recall,
    'F1-Score': baseline_f1,
    'AUC-ROC': baseline_auc
}

print("Baseline Metrics:\n", baseline_metrics)

# Baseline scores
baseline_scores = [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1, baseline_auc]

# Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
knn_scores = [knn_results[0], knn_results[1], knn_results[2], knn_results[3], knn_results[4]]
mlp_scores = [mlp_results[0], mlp_results[1], mlp_results[2], mlp_results[3], mlp_results[4]]
svm_scores = [svm_results[0], svm_results[1], svm_results[2], svm_results[3], svm_results[4]]

x = range(len(metrics))

plt.figure(figsize=(12, 6))
plt.bar(x, baseline_scores, width=0.2, label='Baseline', align='center')
plt.bar([p + 0.2 for p in x], knn_scores, width=0.2, label='KNN', align='center')
plt.bar([p + 0.4 for p in x], mlp_scores, width=0.2, label='MLP', align='center')
plt.bar([p + 0.6 for p in x], svm_scores, width=0.2, label='SVM', align='center')
plt.xticks([p + 0.3 for p in x], metrics)
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Comparison of Classifier Performance with Baseline')
plt.legend(loc='upper left')
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 6))

# KNN
knn_best_model = knn_grid_search.best_estimator_
y_scores_knn = knn_best_model.predict_proba(X_test_scaled)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_scores_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')

# MLP
mlp_best_model = mlp_grid_search.best_estimator_
y_scores_mlp = mlp_best_model.predict_proba(X_test_scaled)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_scores_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC = {roc_auc_mlp:.2f})')

# SVM
svm_best_model = svm_grid_search.best_estimator_
y_scores_svm = svm_best_model.predict_proba(X_test_scaled)[:, 1]
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
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()
