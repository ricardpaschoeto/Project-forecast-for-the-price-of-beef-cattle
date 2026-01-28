
"""Definições de Feature Store (SageMaker) — grupos: mercado, clima, sazonal."""
from dataclasses import dataclass
from typing import List

@dataclass
class FeatureSpec:
    name: str
    dtype: str
    description: str = ""

FEATURE_GROUPS = {
    "market": [
        FeatureSpec("price_spot", "float", "Preço CEPEA/B3"),
        FeatureSpec("price_future", "float", "Contrato futuro B3"),
        FeatureSpec("usd_brl", "float", "Câmbio"),
        FeatureSpec("corn_price", "float", "Milho"),
        FeatureSpec("soy_price", "float", "Soja"),
    ],
    "climate": [
        FeatureSpec("precip", "float", "Precipitação"),
        FeatureSpec("temp_mean", "float", "Temperatura média"),
        FeatureSpec("pdsi", "float", "Índice de seca PDSI"),
        FeatureSpec("elnino_index", "float", "El Niño/La Niña"),
    ],
    "seasonal": [
        FeatureSpec("month", "int", "Mês"),
        FeatureSpec("quarter", "int", "Trimestre"),
        FeatureSpec("season_flag", "int", "Safra/Entressafra"),
    ],
}

# TODO: adicionar client boto3/sagemaker para criar/atualizar Feature Groups
