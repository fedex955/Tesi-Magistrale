from sklearn.model_selection import train_test_split
from modules.logs_handler import *
from modules.rtd_dataset import *
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification,Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime
import os


# Class that rapresents the NLP
class DistillBERT_ST():
    # Constructor for the class 
    def __init__(self,logs_handler:LogsHandler):
            # define logs Handler 
            self.logs_handler = logs_handler


    # -- Load the model and tokenizer 
    def _define_model(self, load_path: str = 'distilbert-base-uncased'):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', load_path))
        # define tokenizer 
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path, local_files_only=True)
        # define model for sequence binary classification 
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path,  local_files_only=True, num_labels=2)



    # -- Control device if Cuda or CPU
    def _control_device(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)



    # -- Split dataset into training and test set 
    def _split_dataset(self, test_size: float = 0.2, random_state: float = 42):
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            self.logs_handler.logs['TEXT'], 
            self.logs_handler.logs['LABEL'], 
            test_size=test_size, 
            random_state=random_state
        )
        return train_texts, test_texts, train_labels, test_labels


    
    # Tokenize the dataset divided into train and test 
    def _tokenization(self,  train_texts, test_texts):
          train_encodings = self.tokenizer(train_texts.tolist(), truncation=True, padding=True)
          test_encodings = self.tokenizer(test_texts.tolist(), truncation=True, padding=True)   
          return train_encodings, test_encodings
    

    # Creation of train and test dataset  
    def _create_train_and_test_dataset_rtd(self,train_encodings, test_encodings, train_labels, test_labels):
          train_dataset = RTD_DATASET(train_encodings, train_labels.tolist())
          test_dataset = RTD_DATASET(test_encodings, test_labels.tolist())
          return train_dataset,test_dataset


    # Method to define the trainer from the model defined 
    def _define_trainer(self,training_args,train_dataset,test_dataset)->Trainer:
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics = self._compute_metrics
        )
        return trainer 
    

    # Method to save the model and tokenizer defined  
    def _save_model(self, save_path: str):
        model_path = os.path.join(os.path.dirname(__file__), '..', save_path)
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)



    # Method to define logs to predict to the logs handler object 
    def _define_logs_to_predict(self,normal_logs_file_path:str):
         self.logs_handler._define_logs_to_predict(normal_logs_file_path=normal_logs_file_path)



    # Method to predict logs and save to a csv file 
    def _predict_logs(self, output_folder_path:str):
         # Control if folder exists 
         if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
         current_date = datetime.now().strftime('%Y-%m-%d')
         output_file_path = f'{output_folder_path}/Predicted_Logs_{current_date}.csv'
         # Encoding of data 
         predict_encodings = self.tokenizer(self.logs_handler.logs_predict['TEXT'].tolist(), truncation=True, padding=True, return_tensors='pt')
         self.model.eval()
         with torch.no_grad():
            print("Starting Predictions...")
            outputs = self.model(**predict_encodings)
            # Find the position of maximum logit that rapresent the predict class in last dimension of tensor 
            predictions = torch.argmax(outputs.logits, dim=-1)
            self.logs_handler.logs_predict['PREDICTION'] = predictions.numpy()
            self.logs_handler.logs_predict[['FACILITY', 'FIELD_FAB', 'ALL_LOTS', 'DATETIME', 'TEXT', 'PREDICTION']].to_csv(output_file_path, index=False)
            print('End Predictions: saved to 'f'{output_folder_path}/Predicted_Logs_{current_date}.csv')

   
   # Compute metrics for the trainer such as: accuracy, f1, precision, recall
    def _compute_metrics(self,pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
