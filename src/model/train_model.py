# -*- coding: utf-8 -*-
import os
import joblib
import bentoml
from bentoml.io import NumpyNdarray
from datetime import datetime

# data libraries
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# print("Current working directory:", os.getcwd())
# print(f"Current file: {str(Path(__file__))}")


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the data file
X_train_path = os.path.join(script_dir, '../../data/processed/X_train.csv')
X_test_path = os.path.join(script_dir, '../../data/processed/X_test.csv')
y_train_path = os.path.join(script_dir, '../../data/processed/y_train.csv')
y_test_path = os.path.join(script_dir, '../../data/processed/y_test.csv')

# Read the data
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Instantiate the models
svm = SVR()
rf = RandomForestRegressor(n_jobs = -1, random_state=42)
xg = XGBRegressor(random_state=42)

param_grid_svm = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'degree': [2, 3],
    'gamma': ['scale', 'auto']
}

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

param_grid_xg = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 6, 9],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Outer and inner cross-validation loops
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Initiate the inner loop of nested cross validation
gridcvs = {}

for pgrid, reg, name in zip((param_grid_svm, param_grid_rf, param_grid_xg),
                            (svm, rf, xg),
                            ('Support Vector Regressor', 'Random Forest Regressor', 'XGBRegressor')):
    gcv = GridSearchCV(reg, pgrid, cv=inner_cv, refit=True, scoring='neg_mean_squared_error')
    gridcvs[name] = gcv

# Initiate the outer loop of nested cross validation
outer_scores = {}
outer_scores_mean = {}
outer_scores_std = {}

for name, gs in gridcvs.items():
    ti = datetime.now()   
    nested_score = cross_val_score(gs, X_train, y_train, cv=outer_cv, scoring='neg_mean_squared_error')
    outer_scores[name] = nested_score
    outer_scores_mean[name] = nested_score.mean()
    outer_scores_std[name] = nested_score.std()
    tf = datetime.now()
    print(f'Cross validation process for {name} in {tf - ti}')
    print(f'{name}: outer neg mean squared error {nested_score.mean():.2e} +/- {nested_score.std():.2e}')
    
# Choose the best model and retrain it with full train dataset
best_model = max(outer_scores_mean, key=outer_scores_mean.get)
print(f'Best model: {best_model}')
final_rg = gridcvs[best_model]
final_rg.fit(X_train, y_train)
print(f'Best Parameters: {final_rg.best_params_}')

# Print the neg mean squared error and R² score
y_train_pred=final_rg.predict(X_train)
y_test_pred=final_rg.predict(X_test)
train_nmse = - mean_squared_error(y_true=y_train, y_pred=y_train_pred)
test_nmse = - mean_squared_error(y_true=y_test, y_pred=y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Negative Mean Squared Error (NMSE) for train dataset: {train_nmse:.2e}")
print(f"R² Score for train dataset: {train_r2:.2f}")
print(f"Negative Mean Squared Error (NMSE) for test dataset: {test_nmse:.2e}")
print(f"R² Score for test dataset: {test_r2:.2f}")

# test the model on a single observation
test_data = X_test.iloc[[0]]
# print the actual label
print(f"Actual label: {y_test[0]}")
# print the predicted label
print(f"Predicted label: {final_rg.predict(test_data)[0]}")

# Enregistrer le modèle dans le Model Store de BentoML
model_ref = bentoml.sklearn.save_model("admissions", final_rg)
print(f"Modèle enregistré sous : {model_ref}")

# #--Save the trained model to a file
# model_filename = './src/models/trained_model.joblib'
# joblib.dump(rf_regressor, model_filename)
# print("Model trained and saved successfully.")