from modules.distillbert_st import *
from modules.logs_handler import *
from modules.prophet_model import *

def main():
    # Percorsi dei file di log e della cartella di output
    normal_logs_file_path = 'logs/NormalLogs.txt'
    anomalous_logs_file_path = 'logs/AnomoulousLogs.txt'
    logs_to_predict_file_path = 'logs/Logs_To_Predict.txt'
    output_folder_path = 'results_prediction_logs0'

    # Inizializza il gestore dei log
    logs_handler = LogsHandler(normal_logs_file_path, anomalous_logs_file_path)

    # Concatenazione e bilanciamento dei log
    logs_handler._concate_logs()

    # Riduzione delle caratteristiche dei log
    logs_handler._reduce_feature_logs()

    # Inizializza il modello DistillBERT
    distilber_st = DistillBERT_ST(logs_handler)

    # Carica il modello e il tokenizer dal percorso salvato
    distilber_st._define_model(load_path='distilbert-base-uncased')
    distilber_st._control_device()

    # Definisci i log da prevedere
    distilber_st._define_logs_to_predict(normal_logs_file_path=logs_to_predict_file_path)

    # Esegui la previsione e salva i risultati
    distilber_st._predict_logs(output_folder_path=output_folder_path)

    # Previsione dei picchi futuri
#    predict_future_peaks(normal_logs_file_path, output_folder_path, periods=30)

if __name__ == "__main__":
    main()
