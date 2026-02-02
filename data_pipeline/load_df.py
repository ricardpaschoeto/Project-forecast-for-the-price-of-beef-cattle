import pandas as pd
import numpy as np


df = pd.read_csv("D:/Users/Ricar/OneDrive/Documentos/PROJETOS_ML_DATA_SCIENCE/PROJETO_AGRO_GOIAS/PROJETO_01/data_pipeline/dados_agro.csv")
print("DataFrame loaded successfully.")

df.insert(6, "boi_negociado", np.zeros(len(df)), True)

#df.drop(columns=["boi_real.1"], inplace=True)

print(df.shape)

df.to_csv("D:/Users/Ricar/OneDrive/Documentos/PROJETOS_ML_DATA_SCIENCE/PROJETO_AGRO_GOIAS/PROJETO_01/data_pipeline/dados_agro_atualizado.csv")

df_ = pd.read_csv("D:/Users/Ricar/OneDrive/Documentos/PROJETOS_ML_DATA_SCIENCE/PROJETO_AGRO_GOIAS/PROJETO_01/data_pipeline/dados_agro_atualizado.csv", index_col=0)

print(df_.columns.tolist())