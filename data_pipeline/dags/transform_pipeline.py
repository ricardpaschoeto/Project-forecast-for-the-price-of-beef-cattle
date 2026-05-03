from clean_pipeline import clean
from feature_engineering import TimeFeaturesTransformer, FeatureSelection
import logging
import os


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
    
    # Step 2: Apply Feature Engineering
    cleaned_data.reset_index(inplace=True)
    time_fe = TimeFeaturesTransformer('data')
    fe = time_fe.transform(cleaned_data)
    fe.set_index('data', inplace=True)

     # Step 3: Load
    # TODO:insert data in postgres 

    # Step 4: feature Selection
    fe_sel = FeatureSelection(fe, 'boi_negociado')
    final_features, votes = fe_sel.temporal_consensus_selection(lags=12, min_votes=2, top_n=28)
    fe_sel.print_features(final_features, votes)

    # Step 5: Save the data
    # TODO: SELECT FROM postgres
    actual_dir = os.path.dirname(os.path.dirname(__file__))
    output_path_fs = os.path.join(actual_dir, 'sensors','dados_modelo_fs.csv')
    fe[final_features + ['boi_negociado']].to_csv(output_path_fs)

if __name__ == "__main__":
    transform()    
