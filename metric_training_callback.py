from transformers import TrainerCallback
from sklearn.metrics import *
import matplotlib.pyplot as plt
import os

# Define a class to create a custom Callback for the trainer 
class MetricsCallback(TrainerCallback):

    def __init__(self, trainer):
        self.metrics = {
            'epoch': [],
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'loss':[]
        }
        self.trainer = trainer 

    # Method to upgrade metrics: accuracy, f1score, precision and recall in every epoch 
    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = self.trainer.evaluate()
        self.metrics['epoch'].append(state.epoch)
        self.metrics['loss'].append(metrics.get('eval_loss',0))
        self.metrics['accuracy'].append(metrics.get('eval_accuracy', 0))
        self.metrics['f1'].append(metrics.get('eval_f1', 0))
        self.metrics['precision'].append(metrics.get('eval_precision', 0))
        self.metrics['recall'].append(metrics.get('eval_recall', 0))

    
    # Method to plot metrics: accuracy, f1score, precision and recall and save as png images in a specific dir 
    def plot_and_save_metrics(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for metric_name in ['accuracy', 'f1', 'precision', 'recall','loss']:
            plt.figure()
            plt.plot(self.metrics['epoch'], self.metrics[metric_name], label=metric_name)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} over epochs')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'{metric_name}.png'))
            plt.close()
        
        print(f"Metrics saved on : {output_dir}")
