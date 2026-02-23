from clean_pipeline import clean
from normalize_pipeline import normalize_time_series

import os
from pathlib import Path

caminho = Path(os.path.abspath(__file__))
transdorm_dir = caminho.parent


def transform():
    # Step 1: Clean the data
    cleaned_data = clean()
    
    # Step 2: Normalize the time series data
    #normalized_data, report = normalize_time_series(True, cleaned_data, cleaned_data.columns)

    # Step 3: Print the normalization report
    #print(report)

    # Step 4: Select only the columns that were normalized
    #normalized_data_ = normalized_data.loc[:, normalized_data.columns.str.contains('norm')]

    # Step 5: Save the transformed data to a new file
    #output_path = os.path.join(transdorm_dir, 'operatores','transformed_data.csv')
    #normalized_data_.to_csv(output_path, index=True)

    #print(normalized_data_.tail())

if __name__ == "__main__":
    transform()
    
    