import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar dataset original sin eliminar columnas
df = pd.read_csv('jobs_in_data.csv')

# 2. Codificación automática de variables categóricas (one-hot)
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Separar variables predictoras y target
X = df_encoded.drop(columns=["salary_in_usd"])
y = df_encoded["salary_in_usd"]

# 4. División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Modelo base: DecisionTreeRegressor
base_tree = DecisionTreeRegressor(max_depth=20, random_state=42)

# 6. Bagging con 250 árboles (como en Weka)
bagging_model = BaggingRegressor(
    estimator=base_tree,
    n_estimators=250,
    random_state=42,
    n_jobs=-1
)

# 7. Entrenar
bagging_model.fit(X_train, y_train)

# 8. Predicción y evaluación
y_pred = bagging_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)
correlation = np.corrcoef(y_test, y_pred)[0, 1]

# 9. Mostrar resultados
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print(f"Coeficiente de correlación: {correlation:.4f}")


