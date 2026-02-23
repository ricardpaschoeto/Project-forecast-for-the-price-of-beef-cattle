
# ml-pipeline/train.py
import argparse
import json
from dataclasses import dataclass
from typing import List, Optional, Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Normalização (reuso do seu módulo)
from normalize_pipeline import normalize_time_series  # noqa: E402
# └─ irá salvar/usar os artefatos em operators/power_transformer.joblib e operators/scaler.joblib  @dataclass
class TrainConfig:
    experiment: str
    run_name: str
    study_name: str

    train_csv: str
    test_csv: Optional[str]

    date_col: str
    target_col: str

    # colunas a normalizar (ex.: exógenas e/ou demais numéricas já selecionadas)
    norm_cols: List[str]

    # Optuna
    n_trials: int = 40
    n_splits: int = 5
    seed: int = 42


def _prepare_norm(df: pd.DataFrame, *, cfg: TrainConfig, is_train: bool) -> pd.DataFrame:
    """
    Aplica normalize_time_series APENAS nas colunas de interesse (cfg.norm_cols),
    preservando as demais colunas como estão. Gera colunas <col>_norm e retorna
    um DataFrame contendo:
      - date_col
      - target_col
      - <col>_norm para cada col em norm_cols
    """
    # normaliza apenas as colunas informadas (já limpas/transformadas/selecionadas)
    df_norm, _ = normalize_time_series(
        isTrain=is_train,
        df=df,
        cols=cfg.norm_cols,
        config=None
    )  # artefatos (scaler/pt) serão salvos/carregados do diretório operators/  [1](https://etnweb-my.sharepoint.com/personal/rmoreir_eletronuclear_gov_br/Documents/Arquivos%20de%20Chat%20do%20Microsoft%20Copilot/normalize_pipeline.py)

    # constrói a matriz final usando apenas as colunas _norm
    keep_cols = [cfg.date_col, cfg.target_col] + [f"{c}_norm" for c in cfg.norm_cols if f"{c}_norm" in df_norm.columns]
    missing = [c for c in keep_cols if c not in df_norm.columns]
    if missing:
        raise ValueError(f"Colunas esperadas não encontradas após normalização: {missing}")

    return df_norm[keep_cols].copy()


def _build_model(trial: optuna.Trial) -> HistGradientBoostingRegressor:
    """
    Espaço de busca Optuna para HGBR.
    """
    params = dict(
        max_depth=trial.suggest_int('max_depth', 3, 12),
        learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        max_iter=trial.suggest_int('max_iter', 200, 1200),
        l2_regularization=trial.suggest_float('l2_regularization', 0.0, 1.0),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 60),
        random_state=42,
    )
    return HistGradientBoostingRegressor(**params)


def _objective(trial: optuna.Trial, cfg: TrainConfig, df_train_in: pd.DataFrame) -> float:
    """
    Otimiza HGBR com TimeSeriesSplit em cima do TREINO normalizado.
    """
    # normaliza (treino)
    df_train = _prepare_norm(df_train_in, cfg=cfg, is_train=True)

    y = df_train[cfg.target_col].astype(float).values
    X = df_train[[f"{c}_norm" for c in cfg.norm_cols]].values

    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    maes = []
    model = _build_model(trial)

    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, preds))

    score = float(np.mean(maes))
    mlflow.log_metric("cv_mae", score)
    # log subset de hiperparâmetros desta iteração
    mlflow.log_params({k: v for k, v in model.get_params().items()
                       if k in {'max_depth','learning_rate','max_iter','l2_regularization','min_samples_leaf'}})
    return score


def main(args):
    cfg = TrainConfig(
        experiment=args.experiment,
        run_name=args.run_name,
        study_name=args.study_name,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        date_col=args.date_col,
        target_col=args.target_col,
        norm_cols=[c.strip() for c in args.norm_cols.split(',') if c.strip()],
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    mlflow.set_experiment(cfg.experiment)

    # Carrega CSVs já limpos/transformados/feature-selected
    df_train_in = pd.read_csv(cfg.train_csv, parse_dates=[cfg.date_col])
    df_test_in = pd.read_csv(cfg.test_csv, parse_dates=[cfg.date_col]) if cfg.test_csv else None

    with mlflow.start_run(run_name=cfg.run_name):
        # Log de configuração
        mlflow.log_param("target_col", cfg.target_col)
        mlflow.log_param("date_col", cfg.date_col)
        mlflow.log_param("norm_cols", cfg.norm_cols)
        mlflow.log_param("model_type", "HGBR_point_scenario")

        # Optuna
        study = optuna.create_study(direction="minimize", study_name=cfg.study_name)
        study.optimize(lambda t: _objective(t, cfg, df_train_in), n_trials=cfg.n_trials, show_progress_bar=True)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_mae", study.best_value)

        # Treina final no treino normalizado com melhores hiperparâmetros
        df_train = _prepare_norm(df_train_in, cfg=cfg, is_train=True)
        y_tr = df_train[cfg.target_col].astype(float).values
        X_tr = df_train[[f"{c}_norm" for c in cfg.norm_cols]].values

        final_model = HistGradientBoostingRegressor(**study.best_params, random_state=42)
        final_model.fit(X_tr, y_tr)

        # Avalia no teste (se fornecido), aplicando normalização com isTrain=False
        if df_test_in is not None:
            df_test = _prepare_norm(df_test_in, cfg=cfg, is_train=False)
            y_te = df_test[cfg.target_col].astype(float).values
            X_te = df_test[[f"{c}_norm" for c in cfg.norm_cols]].values
            preds_te = final_model.predict(X_te)
            test_mae = mean_absolute_error(y_te, preds_te)
            mlflow.log_metric("test_mae", float(test_mae))

        # Loga o estimador para MLflow
        mlflow.sklearn.log_model(final_model, artifact_path="model")
        print(f"[Optuna] Best CV MAE: {study.best_value:.4f}")
        if df_test_in is not None:
            print(f"[Test] MAE: {test_mae:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=str, default='boi-gordo')
    p.add_argument('--run_name', type=str, default='HGBR_point_scenario_optuna')
    p.add_argument('--study_name', type=str, default='optuna_point_scenario')

    p.add_argument('--train_csv', type=str, required=True)
    p.add_argument('--test_csv', type=str, default='')

    p.add_argument('--date_col', type=str, default='Date')
    p.add_argument('--target_col', type=str, default='preco_arroba')

    # As colunas que serão normalizadas pelo normalize_pipeline (limpas/transformadas/selecionadas previamente)
    p.add_argument('--norm_cols', type=str, required=True, help='lista separada por vírgula, ex.: "selic,ipca,cambio"')

    p.add_argument('--n_trials', type=int, default=40)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)

    args = p.parse_args()
    main(args)
