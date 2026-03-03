from clean_pipeline import clean
from feature_engineering import TimeFeaturesTransformer, FeatureSelection
import logging

import pandas as pd

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt='[%(levelname)s] %(name)s - %(message)s'
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

def transform():

    # Step1 : Extract data from web
    # TODO: code extraction 
    
    # Step 1: Clean the data
    cleaned_data = clean()

    # Step 2: load data
    # TODO:insert data in postgres 
    
    # Step 2: Apply Feature Engineering
    cleaned_data.reset_index(inplace=True)
    time_fe = TimeFeaturesTransformer('data')
    fe = time_fe.transform(cleaned_data)
    fe.set_index('data', inplace=True)

    # Step 3: feature Selection
    fe_sel = FeatureSelection(fe, 'boi_negociado')
    selected_rfe, selected_lasso = fe_sel.display_selected_features()

    # Step 4: Save the data
    actual_dir = os.path.dirname(os.path.dirname(__file__))
    output_path_lasso = os.path.join(actual_dir, 'sensors','dados_modelo_lasso.csv')
    output_path_rfe = os.path.join(actual_dir, 'sensors','dados_modelo_rfe.csv')
    fe[selected_lasso + ['boi_negociado']].to_csv(output_path_lasso)
    fe[selected_rfe + ['boi_negociado']].to_csv(output_path_rfe)

if __name__ == "__main__":
    transform()
    
    