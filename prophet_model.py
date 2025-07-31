import pandas as pd
from prophet import Prophet
from datetime import datetime
import os

class Prophet_ST:
    def __init__(self):
        self.logs = None  # Inizializza l'attributo logs

    def load_logs(self, logs_file_path):
        # Carica i dati dal file e assegna a self.logs
        self.logs = pd.read_csv(logs_file_path, delimiter='|', header=None)

    def predict_future_peaks(self, logs_file_path, output_folder_path, periods=30):
        self.load_logs(logs_file_path)  # Carica i dati nei logs
        print(f"logs_file_path: {logs_file_path}, output_folder_path: {output_folder_path}, periods: {periods}")

        # Verifica il numero di colonne nel DataFrame
        expected_columns = ['FACILITY', 'FIELD_FAB', 'ALL_LOTS', 'RULE', 'FIELD_NULL', 'FIELD_0', 
                            'TIMESTAMP', 'DATETIME', 'FIELD1', 'FIELD2', 'FIELD3', 
                            'FIELD4', 'FIELD5', 'FIELD6', 'FIELD7', 'EXECUTION_TIME', 'HOST', 
                            'FIELD10', 'FIELD11', 'FIELD12', 'FIELD13', 'FIELD14', 'FIELD15', 
                            'FIELD16', 'FIELD17', 'FIELD18', 'FIELD19', 'FIELD20', 'FIELD21']
        
        if self.logs.shape[1] == len(expected_columns):
            self.logs.columns = expected_columns
        else:
            raise ValueError(f"Length mismatch: Expected axis has {self.logs.shape[1]} elements, new values have {len(expected_columns)} elements")
        
        # Prepara i dati per Prophet
        self.logs['DATETIME'] = pd.to_datetime(self.logs['DATETIME'].str.strip(), format='%d/%m/%Y %H:%M:%S.%f%z')  # Specifica il formato della data e ora
        logs_for_prophet = self.logs[['DATETIME', 'EXECUTION_TIME']]
        logs_for_prophet.columns = ['ds', 'y']

        # Inizializza e addestra il modello Prophet
        model = Prophet()
        model.fit(logs_for_prophet)

        # Crea un dataframe per le previsioni future
        future = model.make_future_dataframe(periods=periods)  # Prevedi i prossimi 'periods' giorni

        # Fai le previsioni
        forecast = model.predict(future)

        # Visualizza le previsioni
        model.plot(forecast)
        model.plot_components(forecast)

        # Salva le previsioni in un file CSV
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        output_file_path = os.path.join(output_folder_path, f'Predicted_Logs_{current_date}.csv')
        forecast.to_csv(output_file_path, index=True)
