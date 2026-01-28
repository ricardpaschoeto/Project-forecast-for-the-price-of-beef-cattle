
import argparse
import os
import time
import mlflow
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Stub de treino: gera dados sintéticos para fins de esqueleto

def main(args):
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "ridge")
        mlflow.log_param("window", args.window)

        # dados artificiais
        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 5))
        y = X @ np.array([0.5, -0.2, 0.1, 0.0, 0.7]) + rng.normal(scale=0.2, size=500)

        # split temporal simples
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = Ridge(alpha=1.0).fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")
        print(f"Run ID: {run.info.run_id} | MAE={mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='boi-gordo')
    parser.add_argument('--window', type=int, default=30)
    args = parser.parse_args()
    main(args)
