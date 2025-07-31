import pandas as pd

# Class to handle log file 
class LogsHandler():
    # -- This constructor load normal and anomoulous log from two different files 
    def __init__(self,normal_logs_file_path:str, anomalous_logs_file_path:str,):
        self.normal_logs = pd.read_csv(normal_logs_file_path, delimiter='|', header=None)
        self.anomalous_logs = pd.read_csv(anomalous_logs_file_path, delimiter='|', header=None)
        self.anomalous_logs['LABEL'] = 1  # Anomalous
        self.normal_logs['LABEL'] = 0  # Normals

    # -- method to concate logs 
    def _concate_logs(self): 
        normal_logs_sampled = self.normal_logs
        if(len(self.anomalous_logs)< len(self.normal_logs)):
            normal_logs_sampled = self.normal_logs.sample(n=len(self.anomalous_logs), random_state=42)
        # sample of normal logs 
        self.logs = pd.concat([self.anomalous_logs, normal_logs_sampled], ignore_index=True)


    # -- method to reduce the number of columns of logs
    def _reduce_feature_logs(self):
        # rename columns 
        self.logs.columns = ['FACILITY', 'FIELD_FAB', 'ALL_LOTS', 'RULE', 'FIELD_NULL', 'FIELD_0', 
                        'TIMESTAMP', 'DATETIME', 'FIELD1', 'FIELD2', 'FIELD3', 
                        'FIELD4', 'FIELD5', 'FIELD6', 'FIELD7', 'EXECUTION_TIME', 'HOST', 
                        'FIELD10', 'FIELD11', 'FIELD12', 'FIELD13', 'FIELD14', 'FIELD15', 
                        'FIELD16', 'FIELD17', 'FIELD18', 'FIELD19', 'FIELD20', 'FIELD21', 
                        'LABEL']

        # Select only features column to consider for the classification 
        self.logs = self.logs[['RULE', 'EXECUTION_TIME', 'HOST', 'LABEL']]
        
        # Compact the columns in a single row for every log to perform 
        self.logs['TEXT'] = self.logs.apply(lambda row: f"{row['RULE']} {row['EXECUTION_TIME']} {row['HOST']}", axis=1)
        
        # Select only compact logs and labels 
        self.logs = self.logs[['TEXT', 'LABEL']] 



    # Method to define logs to predict 
    def _define_logs_to_predict(self, normal_logs_file_path:str):
        self.logs_predict = pd.read_csv(normal_logs_file_path, delimiter='|', header=None)
        # rename columns 
        self.logs_predict.columns = ['FACILITY', 'FIELD_FAB', 'ALL_LOTS', 'RULE', 'FIELD_NULL', 'FIELD_0', 
                        'TIMESTAMP', 'DATETIME', 'FIELD1', 'FIELD2', 'FIELD3', 
                        'FIELD4', 'FIELD5', 'FIELD6', 'FIELD7', 'EXECUTION_TIME', 'HOST', 
                        'FIELD10', 'FIELD11', 'FIELD12', 'FIELD13', 'FIELD14', 'FIELD15', 
                        'FIELD16', 'FIELD17', 'FIELD18', 'FIELD19', 'FIELD20', 'FIELD21']
        self.logs_predict = self.logs_predict[['FACILITY', 'FIELD_FAB', 'ALL_LOTS', 'DATETIME','RULE', 'EXECUTION_TIME', 'HOST']]
        # Select only features column to consider for the classification 
        self.logs_predict['TEXT'] = self.logs_predict.apply(lambda row: f"{row['RULE']} {row['EXECUTION_TIME']} {row['HOST']}", axis=1)
