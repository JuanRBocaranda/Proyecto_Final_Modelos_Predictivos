# Proyecto Final – Modelos Predictivos

**Predicción de salario en el campo de datos**

## 1. Base de datos

Se utilizó el dataset `jobs_in_data.csv`, el cual contiene 9,355 registros de personas empleadas en el ámbito de datos. Los registros cubren diferentes roles, niveles de experiencia, tipos de empleo y ubicaciones geográficas. Esta base de datos fue seleccionada por su diversidad de variables categóricas y el hecho de que el salario está normalizado en USD (`salary_in_usd`).

El archivo original fue descargado desde Kaggle y se mantuvo sin filtrado para el modelo final, ya que se comprobó que usar toda la información aumentaba la capacidad predictiva.

## 2. Descripción de las columnas del dataset

| Columna              | Descripción                                                                 |
|----------------------|------------------------------------------------------------------------------|
| `work_year`          | Año del registro salarial (2020 a 2023)                                     |
| `job_title`          | Título específico del puesto laboral                                       |
| `job_category`       | Categoría general del rol                                                   |
| `salary_currency`    | Moneda original del salario                                                  |
| `salary`             | Salario reportado en moneda original                                        |
| `salary_in_usd`      | Salario convertido a dólares estadounidenses                               |
| `employee_residence` | País de residencia del empleado                                             |
| `experience_level`   | Nivel de experiencia (Entry, Mid, Senior, Executive)                        |
| `employment_type`    | Tipo de empleo (Full-time, Part-time, Freelance, Contract)                  |
| `work_setting`       | Modalidad de trabajo (Remote, In-person, Hybrid)                            |
| `company_location`   | Ubicación de la empresa                                                      |
| `company_size`       | Tamaño de la empresa (S, M, L)                                               |

## 3. Análisis descriptivo

Se llevó a cabo un análisis exploratorio de datos (EDA) con los siguientes pasos:

- Carga del dataset en pandas:

```python
import pandas as pd
df = pd.read_csv("jobs_in_data.csv")

# Información general del DataFrame
df.info()

# Primeras filas del dataset
df.head()
```

- Visualización de distribución de salarios:

```python
import matplotlib.pyplot as plt

df["salary_in_usd"].plot(kind="hist", bins=30, color="skyblue")
plt.title("Distribución de salarios en USD")
plt.xlabel("Salario")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```
![histograma_salario](https://github.com/user-attachments/assets/078cf04a-068e-4ee1-a2cb-6732f44a50f3)

**Resultado esperado:** Un histograma con asimetría hacia la derecha, mostrando mayor frecuencia de salarios entre 40,000 y 150,000 USD.

- Boxplot por nivel de experiencia:

```python
import seaborn as sns

sns.boxplot(x="experience_level", y="salary_in_usd", data=df)
plt.title("Salario por nivel de experiencia")
plt.show()
```
![boxplot_experiencia_salario](https://github.com/user-attachments/assets/81ce269d-927a-4020-9354-8f70d786d480)

**Resultado esperado:** Incremento progresivo del salario conforme aumenta el nivel de experiencia (`Entry < Mid < Senior < Executive`).

- Gráfico de barras por categoría laboral:

```python
df["job_category"].value_counts().plot(kind="bar")
plt.title("Distribución por categoría laboral")
plt.xticks(rotation=45)
plt.show()
```
![barras_categoria_laboral](https://github.com/user-attachments/assets/b809a8ea-5bab-45ba-ae4b-e26a04f8b6bd)

**Resultado esperado:** Mayor presencia de registros en `Data Science and Research`, seguido por `Data Analysis` y `Data Engineering`.

## 4. Análisis predictivo

Se realizaron análisis predictivos tanto en **Python** como en **Weka** para comparar el rendimiento de diferentes modelos aplicados al mismo conjunto de datos.

### Uso de Weka:

En Weka se aplicó el algoritmo **RandomForest** con dos configuraciones distintas:

1. Dataset filtrado (solo empleados en EE. UU., full-time, con empresa en EE. UU.)
2. Dataset completo (sin filtrado por residencia ni tipo de empleo)

En ambos casos se utilizó como variable objetivo `salary_in_usd` y todas las demás variables categóricas se mantuvieron. Weka realiza automáticamente la codificación de atributos nominales.

**Resultados en Weka:**
- Dataset filtrado: Correlación ≈ 0.4906, RMSE ≈ 51,516 USD
- Dataset completo: Correlación ≈ 0.9837, RMSE ≈ 11,666 USD

Se concluyó que el modelo con dataset completo era claramente superior, por lo que se decidió replicar este enfoque en Python para obtener mayor control del preprocesamiento y explorar más modelos.

### Preprocesamiento en Python:

- Codificación de variables categóricas:

```python
df_encoded = pd.get_dummies(df, drop_first=True)
```

- División de variables predictoras y objetivo:

```python
X = df_encoded.drop(columns=["salary_in_usd"])
y = df_encoded["salary_in_usd"]
```

- División en entrenamiento y prueba:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Entrenamiento del modelo final:

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

model = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=20),
    n_estimators=250,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
correlation = np.corrcoef(y_test, y_pred)[0, 1]

print("RMSE:", rmse)
print("R^2:", r2)
print("Correlación:", correlation)
```
**Resultados obtenidos:**
```
RMSE: 7459.49
R^2: 0.9863
Correlación: 0.9931
```

Estos resultados indican una altísima capacidad predictiva del modelo final.

## 5. Archivos del proyecto

- `jobs_in_data.csv`: dataset completo
- `analisis_descriptivo.py`: script de visualización de datos
- `analisis_predictivo.py`: script de entrenamiento del modelo Bagging + DecisionTree
- `weka_modelos/`: capturas y archivos `.arff` utilizados en Weka

