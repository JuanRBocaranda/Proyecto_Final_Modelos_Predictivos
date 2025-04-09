import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset original
df = pd.read_csv('jobs_in_data.csv')

# Información general del DataFrame
df.info()

# Primeras filas del dataset
df.head()

# === Gráfico 1: Histograma del salario ===
plt.figure(figsize=(8, 5))
df["salary_in_usd"].plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
plt.title("Distribución del Salario en USD")
plt.xlabel("Salario (USD)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfico 2: Boxplot del salario por nivel de experiencia ===
plt.figure(figsize=(8, 5))
sns.boxplot(x="experience_level", y="salary_in_usd", data=df)
plt.title("Salario por Nivel de Experiencia")
plt.xlabel("Nivel de Experiencia")
plt.ylabel("Salario (USD)")
plt.tight_layout()
plt.show()

# === Gráfico 3: Barras - cantidad por categoría laboral ===
plt.figure(figsize=(10, 5))
df["job_category"].value_counts().plot(kind='bar', color='lightgreen')
plt.title("Distribución de Categorías Laborales")
plt.xlabel("Categoría")
plt.ylabel("Cantidad de Registros")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



