
import numpy as np
import pandas as pd 
import os
import joblib
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

print(joblib.__version__)

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



# X_train = pd.read_csv('./data/processed/X_train.csv')
# X_test = pd.read_csv('./data/processed/X_test.csv')
# y_train = pd.read_csv('./data/processed/y_train.csv')
# y_test = pd.read_csv('./data/processed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

rf_regressor = ensemble.RandomForestRegressor(n_jobs = -1)

#--Train the model
rf_regressor.fit(X_train, y_train)

#--Test the model
y_pred = rf_regressor.predict(X_test)

#--Get the model score
mse = mean_squared_error(y_test, y_pred)

# Optionally, calculate R² score as well
r2 = rf_regressor.score(X_test, y_test)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# test the model on a single observation
test_data = X_test.iloc[[0]]
# print the actual label
print(f"Actual label: {y_test[0]}")
# print the predicted label
print(f"Predicted label: {rf_regressor.predict(test_data)[0]}")

# Enregistrer le modèle dans le Model Store de BentoML
model_ref = bentoml.sklearn.save_model("admissions", rf_regressor)

print(f"Modèle enregistré sous : {model_ref}")


# #--Save the trained model to a file
# model_filename = './src/models/trained_model.joblib'
# joblib.dump(rf_regressor, model_filename)
# print("Model trained and saved successfully.")