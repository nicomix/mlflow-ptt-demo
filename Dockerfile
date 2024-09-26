# Utiliza una imagen base oficial de Python
FROM python:3.10

# Instala MLflow y otras dependencias
RUN pip install mlflow torch torchvision

# Expone el puerto para acceder al servidor MLflow
EXPOSE 5000

# Comando para iniciar MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
