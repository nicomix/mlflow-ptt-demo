import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

# Definir un modelo de red neuronal simple
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función para entrenar el modelo
def train_model():
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Datos de ejemplo
    inputs = torch.randn(10, 4)
    targets = torch.randn(10, 2)

    # Configuración de MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("pytorch-demo")

    with mlflow.start_run():
        mlflow.log_param("lr", 0.01)
        mlflow.log_param("optimizer", "Adam")

        # Entrenamiento
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                mlflow.log_metric("loss", loss.item(), step=epoch)

        # Guardar el modelo entrenado
        mlflow.pytorch.log_model(model, "model")

        # Guardar los pesos del modelo
        torch.save(model.state_dict(), "model_weights.pth")
        mlflow.log_artifact("model_weights.pth")

if __name__ == "__main__":
    train_model()
