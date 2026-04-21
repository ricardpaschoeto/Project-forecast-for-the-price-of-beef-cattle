from pathlib import Path

import pandas as pd
import os

def _load_history():
    path = Path(os.path.abspath(__file__))
    root_dir = path.parent.parent
    df_path = os.path.join(root_dir,'data_pipeline', 'sensors', 'dados_modelo_rfe.csv')
    df = pd.read_csv(df_path, index_col=0, dayfirst=True)
    df.index = pd.to_datetime(df.index, dayfirst=True)
    return df

def load_cenarios(dates: list[str]):
    scenario = {}
    list_scenarios = []
    for date in dates:
        scenario[date] = _load_history().loc[date].to_dict()
        list_scenarios.append(scenario[date])

    return list_scenarios

