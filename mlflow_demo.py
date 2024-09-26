import numpy as np
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Configurar la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Leer el dataset desde el archivo CSV
df = pd.read_csv('Ice_cream selling data.csv')

# Separar características (X) y etiquetas (y)
X = df[['Temperature (°C)']].values
y = df['Ice Cream Sales (units)'].values

# Verificar si hay valores NaN o infinitos en los datos
assert not np.any(np.isnan(X)), "X contiene valores NaN"
assert not np.any(np.isnan(y)), "y contiene valores NaN"
assert not np.any(np.isinf(X)), "X contiene valores infinitos"
assert not np.any(np.isinf(y)), "y contiene valores infinitos"

# Generar características polinómicas
degree = 3  # Cambia el grado si lo prefieres
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Normalizar los datos
scaler = StandardScaler()
X_poly = scaler.fit_transform(X_poly)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Configurar el experimento MLflow
mlflow.set_experiment("Ice Cream Sales Prediction")

with mlflow.start_run():
    # Loggear parámetros adicionales
    mlflow.log_param("degree", degree)  # Loggear el grado del polinomio
    mlflow.log_param("model_name", "Ridge Regression")  # Loggear el nombre del modelo
    mlflow.log_param("alpha", 1.0)  # Loggear el parámetro de regularización

    # Entrenar el modelo de regresión polinómica con regularización L2
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    test_loss = mean_squared_error(y_test, y_pred)
    print(f'Test Loss: {test_loss:.4f}')

    # Loggear métricas en MLflow
    mlflow.log_metric("test_loss", test_loss)

    # Generar y loggear el scatter plot con la línea de predicción
    y_pred_all = model.predict(X_poly)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color='blue', label='Actual Data')
    plt.plot(X[:, 0], y_pred_all, color='red', label='Predicted Line')
    plt.title('Ice Cream Sales Prediction')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Ice Cream Sales (units)')
    plt.legend()
    plt.savefig('scatter_plot_with_predictions.png')
    plt.close()

    # Loggear el scatter plot con la línea de predicción como artifact
    mlflow.log_artifact('scatter_plot_with_predictions.png')

    # Guardar el modelo
    mlflow.sklearn.log_model(model, "model")

print("Modelo entrenado y registrado en MLflow, gráficos loggeados como artifacts.")