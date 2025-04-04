import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from datasets import Dataset
from text_custom_functions import tokenize_function, safe_loader, safe_saver
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def evaluate_model():

    test = pd.read_csv('data/processed/text_test_set.csv')
    test_dataset = Dataset.from_pandas(test)

    test_dataset = test_dataset.map(tokenize_function, batched = True,
                                    num_proc = 4)
    test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                           num_labels = 27)
    model.load_state_dict(torch.load('models/roBERTa_model_weights.pth',
                                     weights_only = True))
    model = model.to(device)

    model.eval()

    test_preds = []
    test_labels = []

    for batch in test_loader:

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask = attention_mask)
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim = 1).detach().cpu().numpy()
        labels = labels.cpu().numpy()

        test_labels.extend(labels)
        test_preds.extend(predicted_classes)

    test_f1 = f1_score(test_labels, test_preds, average = 'weighted')
    test_report = classification_report(test_labels, test_preds, output_dict = True)
        
    safe_saver(test_report, 'metrics/roBERTa_classification_report.pkl')
    history = safe_loader('metrics/roBERTa_performance.pkl')

    labels = ['Training F1-Score', 'Validation F1-Score', 'Test F1-Score']
    scores = [max(history['f1']), max(history['val_f1']), test_f1]

    fig = plt.figure(figsize = (20, 10))

    bars = plt.bar(labels, scores, color = ['blue', 'purple', 'green'])

    for bar in bars:
        yval = round(bar.get_height(), 3)
        plt.text(bar.get_x() + bar.get_width() /2, yval - 0.05, str(yval),
                 ha = 'center', va = 'bottom', color = 'white', fontweight = 'bold')

    plt.ylabel('F1-Score')
    plt.title('Training, Validation, and Text F1-Scores')
    plt.tight_layout()

    plt.savefig('images/roBERTa_f1_scores.png')

    fig, ax = plt.subplots(1, 3, figsize = (20, 5))

    titles = ['Loss', 'F1-Score', 'Gradient']
    training_items = ['loss', 'f1', 'gradient']
    validation_items = ['val_loss', 'val_f1']

    ticks = [1]
    for tick in range(1, len(history['loss'])  + 1):
        if tick % 5 == 0:
            ticks.append(tick)

    for index, axes in enumerate(ax.flat):

        if index == 2:
            axes.plot(range(1, len(history[training_items[index]]) + 1),
                            history[training_items[index]], label = 'Training')
            axes.set_xticks(ticks)
            axes.set_title(f"{titles[index]} by Epoch")
            axes.set_xlabel('Epoch')
            axes.set_ylabel(f"{titles[index]}")
            break

        axes.plot(range(1, len(history[training_items[index]]) + 1),
                        history[training_items[index]], label = 'Training')
        axes.plot(range(1, len(history[validation_items[index]]) + 1),
                        history[validation_items[index]], label = 'Validation')
        axes.set_xticks(ticks)
        axes.set_title(f"{titles[index]} by Epoch")
        axes.set_xlabel('Epoch')
        axes.set_ylabel(f"{titles[index]}")
        axes.legend()

    plt.tight_layout()
    plt.savefig('images/roBERTa_training_history.png')
    

if __name__ == '__main__':
    evaluate_model()
