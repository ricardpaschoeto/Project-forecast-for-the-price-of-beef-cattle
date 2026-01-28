import pandas as pd
import numpy as np

## Clean Data and Feature Engineering
def _clean_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing values and removing duplicates.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.
    cols (list): List of columns to check for missing values.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """

    # Fill missing values with the mean of the column
    df.dropna(subset=columns, inplace=True)

    if df[columns].isna().sum().sum() == 0:
        print("All missing values have been filled.")
    else:
        print("There are still missing values in the DataFrame.")

    return df

def _fill_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Fills missing values in a specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): The columns names where missing values need to be filled.

    Returns:
    pd.DataFrame: The DataFrame with filled missing values.
    """

    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df.set_index("data", inplace=True)

    for col in columns:
        df[col] = df[col].ffill()

    if df[columns].isna().sum().sum() == 0:
        print("All missing values have been filled.")
    else:
        print("There are still missing values in the DataFrame.")

    return df

def _categorical_columns(df: pd.DataFrame, columns: list, ordinal: bool) -> pd.DataFrame:
    """
    Identifies categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): The categorical's columns.

    Returns:
    pd.DataFrame: The DataFrame with encoded categoricals.
    """
    if ordinal:
        # ordinal mapping
        map_force = {
            'INEXISTENTE': 0,
            'FRACO': 1,
            'MÉDIO': 2,
            'FORTE': 3
        }
        try:
            df[str(columns) + '_encoded'] = df[columns].map(map_force)
            df.drop(columns=[columns], inplace=True)
        except Exception as e:
            print(f"Error during ordinal encoding: {e}")
    else:
        try:
            df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
        except Exception as e:
            print(f"Error during one-hot encoding: {e}")

    return df

def clean_pipeline() -> pd.DataFrame:
    """
    Initialize the pipeline.

    Parameters:

    Returns:
    pd.DataFrame: The DataFrame cleaned.
    """
    df = pd.read_csv('dados_agro.csv')

    cols_int64 = ['CovidPeriodFlag',
                'ano_novo_flag',
                'carnaval_core_flag',
                'carnaval_window_flag',
                'pascoa_flag',
                'pascoa_window_flag',
                'dias_maes_flag',
                'dias_maes_weekend_flag',
                'festas_juninas_flag',
                'santo_antonio_flag',
                'sao_joao_flag',
                'sao_pedro_flag',
                'ferias_midyear_flag',
                'ferias_verao_flag',
                'dias_pais_flag',
                'dias_pais_weekend_flag',
                'independencia_flag',
                'independencia_window_flag',
                'dia_criancas_flag',
                'finados_flag',
                'finados_weekend_flag',
                'natal_flag',
                'natal_window_flag',
                'fim_ano_window_flag']    
    df_clean = _clean_data(df, cols_int64)

    cols_mixed = ['el nino', 'precip_total_mm', 'TEMPERATURA', 'ipca', 'selic', 'pib_brasil', 'pib_agro']
    df_clean = _clean_data(df_clean, cols_mixed)

    cols_time_series = ['cme', 'boi_futuro', 'boi_real', 'boi_dolar', 'soja_real', 'soja_dolar', 'soja_futuro',
        'milho_futuro', 'milho_real', 'milho_dolar', 'dolar']    
    df_filled = _fill_data(df_clean, cols_time_series)

    cols_cat_01 = ['ciclo','aspecto']
    cols_cat_02 = 'el nino'
    df_final = _categorical_columns(df_filled, cols_cat_01, ordinal=False)
    df_final = _categorical_columns(df_final, cols_cat_02, ordinal=True)

    df_final.to_csv('cleaned_df.csv', index=True)

if __name__ == "__main__":
    clean_pipeline()