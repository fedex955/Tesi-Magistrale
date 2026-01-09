from modules.RandomForestClassifierModel import *
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

    rf_model = RandomForestClassifierModel(logs_handler)
    train_texts, test_texts, train_labels, test_labels = rf_model._split_dataset()
    train_vectors, test_vectors, vectorizer = rf_model._vectorize_texts(train_texts, test_texts)

    rf_model.train(train_vectors, train_labels)
    results = rf_model.evaluate(test_vectors, test_labels)
    print(results)

    logs_handler._define_logs_to_predict(normal_logs_file_path='logs/Logs_To_Predict.txt')
    rf_model.predict(vectorizer, output_folder_path='results_prediction_logs')
    

if __name__ == "__main__":
    main()
