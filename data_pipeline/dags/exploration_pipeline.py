import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.copy_on_write = True

def _plot_corr_target(df, target, save_path = None):

    def corr_range(corr):
        if np.abs(corr) > 0.2 and corr < 0.9:
            return corr
            
    df = df.drop(columns=['data'])

    correlations = df.corr()[target].drop(target).apply(corr_range)

    correlations = correlations.dropna()

    colors = sns.diverging_palette(10, 130, as_cmap = True)
    
    color_mapped = correlations.map(colors)

    sns.set_style("darkgrid", {"axes.facecolor": "#eaeaf2", "grid.linewidth": 1.5})

    fig = plt.figure(figsize = (12, 8))

    plt.barh(correlations.index, correlations.values, color = color_mapped)

    # Define the title and the axis labels of the chart
    plt.title("Correlação com boi real", fontsize = 16)
    plt.xlabel("Coeficiente de Correlação", fontsize = 14)
    plt.ylabel("Variável", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.grid(axis="x")

    # Adjust the layout to avoid overlapping
    plt.tight_layout()

    # Save the chart as a PNG file, if a path is provided
    if save_path:
        plt.savefig(save_path, format = "png", dpi = 600)

    # Close the figure to free up memory
    plt.close(fig)

def _heatmap(df: pd.DataFrame, save_path: str = None):
    """
    Plots a heatmap of the correlation matrix of the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    save_path (str): The path to save the heatmap image.

    Returns:
    plt.Figure: The heatmap figure.
    """
    corr = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, annot_kws={"size": 8})
    plt.title("Heatmap of Correlation Matrix", fontsize=16)

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    fig = plt.gcf()
    plt.close()

def _heatmap_range(df: pd.DataFrame, save_path: str = None) -> list:
    """
    Plots a heatmap of the correlation matrix of the DataFrame for correlations within a specific range.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    save_path (str): The path to save the heatmap image.

    Returns:
    plt.Figure: The heatmap figure.
    """
    corr = df.corr()

    # Filter correlations within the specified range
    mask = (corr.abs() >= 0.2) & (corr.abs() <= 0.9)
    filtered_corr = corr.where(mask)

    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, annot_kws={"size": 8})
    plt.title("Heatmap of Correlation Matrix (Filtered)", fontsize=16)

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close()

    return filtered_corr[filtered_corr.notna().any(axis=1)].index.tolist()

def transform_pipeline() -> pd.DataFrame:
    """
    Initialize the transformation pipeline.

    Parameters:

    Returns:
    pd.DataFrame: The DataFrame transformed.
    """
    try:
        df_agro = pd.read_csv(r'D:\Users\Ricar\OneDrive\Documentos\PROJETOS_ML_DATA_SCIENCE\PROJETO_AGRO_GOIAS\PROJETO_01\data-pipeline\utils\cleaned_df.csv')
        _plot_corr_target(df_agro, "boi_real", save_path=r'D:\Users\Ricar\OneDrive\Documentos\PROJETOS_ML_DATA_SCIENCE\PROJETO_AGRO_GOIAS\PROJETO_01\data-pipeline\utils\report\correlations_target.png')

        df_agro['data'] = pd.to_datetime(df_agro['data'])
        df_agro.set_index('data', inplace=True)
        
        _heatmap(df_agro, save_path=r'D:\Users\Ricar\OneDrive\Documentos\PROJETOS_ML_DATA_SCIENCE\PROJETO_AGRO_GOIAS\PROJETO_01\data-pipeline\utils\report\heatmap_full_correlation.png')
        filtered_corr = _heatmap_range(df_agro, save_path=r'D:\Users\Ricar\OneDrive\Documentos\PROJETOS_ML_DATA_SCIENCE\PROJETO_AGRO_GOIAS\PROJETO_01\data-pipeline\utils\report\heatmap_highest_correlation.png')

        df_agro[filtered_corr].to_csv('transformed_df.csv', index=False)
    except Exception as e: 
        print(f"An error occurred in transformed_pipeline function: {e}")

if __name__== '__main__':
    transform_pipeline()

