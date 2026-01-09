import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caricare il file CSV locale
file_path = r"C:/Users/crucillf/OneDrive - STMicroelectronics/Documents/Tesi/DistlBERT/results_prediction_logs/Predicted_Logs_2024-08-20.csv"
data = pd.read_csv(file_path)

# Filtrare i primi 20 record
filtered_data = data.head(20)

# Estrarre il "tempo di esecuzione" dalla colonna 'TEXT'
# Supponiamo che il tempo di esecuzione sia la prima parte della stringa separata da uno spazio
filtered_data['Tempo_Esecuzione'] = filtered_data['TEXT'].apply(lambda x: float(x.split()[1]))

# Formattare il tempo di esecuzione per mantenere solo le prime tre cifre dopo il punto
filtered_data['Tempo_Esecuzione'] = filtered_data['Tempo_Esecuzione'].apply(lambda x: f"{x:.3f}")

# Creazione del grafico a barre
plt.figure(figsize=(10, 6))
sns.barplot(x='Tempo_Esecuzione', y='PREDICTION', data=filtered_data, palette='pastel')

# Aggiungere etichette e titolo
plt.xlabel('Tempo di Esecuzione')
plt.ylabel('Predizione')
plt.title('Grafico delle Predizioni')

# Ruotare le etichette dell'asse x se necessario
plt.xticks(rotation=45)

# Salvare il grafico
output_path = r"C:/Users/crucillf/OneDrive - STMicroelectronics/Documents/Tesi/DistlBERT/Plot/plot.png"
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')

# Mostrare il grafico
plt.show()
