import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import os

# Carregar os dados
cwd = os.getcwd()
pathzada = os.path.join(cwd, 'data/datapose_new_and_old.csv')
tpot_data = pd.read_csv(pathzada, sep=',')
tpot_data = tpot_data.drop(['Image_Path', 'Image_Name'], axis=1)
features = tpot_data.drop('Label', axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(features, tpot_data['Label'], random_state=42)

# Definir o pipeline
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    MLPClassifier(max_iter=1000)  # Aumentar max_iter para garantir a convergência
)

# Definir o espaço de hiperparâmetros para explorar
param_grid = {
    'mlpclassifier__alpha': [0.001, 0.01, 0.1],
    'mlpclassifier__learning_rate_init': [0.01, 0.1, 0.2],
    # Você pode adicionar mais parâmetros aqui se necessário
}

# Criar e configurar o GridSearchCV
grid_search = GridSearchCV(exported_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar o modelo usando GridSearchCV
grid_search.fit(training_features, training_target)

# Exibir os melhores parâmetros e o melhor score encontrado
print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor score de validação cruzada:", grid_search.best_score_)

# Salvar o melhor modelo encontrado pelo GridSearchCV
dump(grid_search.best_estimator_, 'mlp_classifier.joblib')

# Usar o melhor modelo para fazer previsões no conjunto de teste
optimized_results = grid_search.predict(testing_features)
print(optimized_results)
