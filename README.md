docker build -t mlflow-demo .
docker run -p 5000:5000 mlflow-demo
python3 mlflow_demo.py