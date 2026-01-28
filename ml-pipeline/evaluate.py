
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Este script ilustra avaliação e logging de métricas (stub)

def main(args):
    client = MlflowClient()
    run = client.get_run(args.run_id)
    # Gera métricas artificiais como exemplo
    y_true = np.linspace(0, 1, 100)
    y_pred = y_true + np.random.normal(scale=0.05, size=100)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    client.log_metric(args.run_id, "rmse", rmse)
    client.log_metric(args.run_id, "mae_eval", mae)
    print(f"Avaliação concluída | RMSE={rmse:.4f} | MAE={mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    main(args)
