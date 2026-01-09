import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime


# Class to handle log file 
class LogsHandler():

    def __init__(self, normal_logs_file_path: str, anomalous_logs_file_path: str):
        self.normal_logs = pd.read_csv(normal_logs_file_path, delimiter='|', header=None)
        self.anomalous_logs = pd.read_csv(anomalous_logs_file_path, delimiter='|', header=None)
        self.anomalous_logs['LABEL'] = 1  # Anomalous
        self.normal_logs['LABEL'] = 0  # Normals

    def _concate_logs(self):
        normal_logs_sampled = self.normal_logs
        if len(self.anomalous_logs) < len(self.normal_logs):
            normal_logs_sampled = self.normal_logs.sample(n=len(self.anomalous_logs), random_state=42)
        self.logs = pd.concat([self.anomalous_logs, normal_logs_sampled], ignore_index=True)

    def _reduce_feature_logs(self):
        self.logs.columns = ['FACILITY', 'FIELD_FAB', 'ALL_LOTS', 'RULE', 'FIELD_NULL', 'FIELD_0', 
                             'TIMESTAMP', 'DATETIME', 'FIELD1', 'FIELD2', 'FIELD3', 
                             'FIELD4', 'FIELD5', 'FIELD6', 'FIELD7', 'EXECUTION_TIME', 'HOST', 
                             'FIELD10', 'FIELD11', 'FIELD12', 'FIELD13', 'FIELD14', 'FIELD15', 
                             'FIELD16', 'FIELD17', 'FIELD18', 'FIELD19', 'FIELD20', 'FIELD21', 
                             'LABEL']
        self.logs = self.logs[['RULE', 'EXECUTION_TIME', 'HOST', 'LABEL']]
        self.logs['TEXT'] = self.logs.apply(lambda row: f"{row['RULE']} {row['EXECUTION_TIME']} {row['HOST']}", axis=1)
        self.logs = self.logs[['TEXT', 'LABEL']]

    def _define_logs_to_predict(self, normal_logs_file_path: str):
        self.logs_predict = pd.read_csv(normal_logs_file_path, delimiter='|', header=None)
        self.logs_predict.columns = ['FACILITY', 'FIELD_FAB', 'ALL_LOTS', 'DATETIME', 'RULE', 'EXECUTION_TIME', 'HOST']
        self.logs_predict['TEXT'] = self.logs_predict.apply(lambda row: f"{row['RULE']} {row['EXECUTION_TIME']} {row['HOST']}", axis=1)

