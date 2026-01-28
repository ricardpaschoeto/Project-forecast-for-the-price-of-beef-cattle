
class PowerBIIntegration:
    def __init__(self, workspace_id: str, dataset_id: str):
        self.workspace_id = workspace_id
        self.dataset_id = dataset_id

    def update_dataset(self, predictions_df):
        # TODO: chamada REST API do Power BI
        print("Atualizando dataset do Power BI — placeholder")

    def trigger_refresh(self):
        print("Disparando refresh do Power BI — placeholder")
