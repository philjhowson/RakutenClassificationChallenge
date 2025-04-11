import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from datasets import Dataset
from text_custom_functions import safe_loader, safe_saver
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

def evaluate_model():

    test = pd.read_csv('data/processed/test_text.csv')
    test_dataset = Dataset.from_pandas(test)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenize_function(dataset):
        return tokenizer(dataset['designation_filtered'], padding = 'max_length',
                     truncation = True, max_length = 128)

    test_dataset = test_dataset.map(tokenize_function, batched = True,
                                    num_proc = 4)
    test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels = 27,
                                                             attn_implementation = "eager")
    model.load_state_dict(torch.load('models/roBERTa_model_weights.pth',
                                     weights_only = True))
    model = model.to(device)

    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()

            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            predicted_classes = torch.argmax(logits, dim = 1).detach().cpu().numpy()

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

    with open('data/processed/test_label_dictionary.json', 'r') as f:
        label_dictionary = json.load(f)

    original_labels = {v: k for k, v in label_dictionary.items()}

    attention_by_label = defaultdict(list)
    tokens_by_label = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            mapped_preds = [original_labels[pred.item()] for pred in preds]

            last_layer_attention = outputs.attentions[-1]
            attention_mean = last_layer_attention.mean(dim=1)

            for i, pred_class in enumerate(mapped_preds):
                attention_by_label[pred_class].append(attention_mean[i].cpu().numpy())
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu())
                tokens = [token for token in tokens if token not in tokenizer.all_special_tokens]
                tokens_by_label[pred_class].append(tokens)


        for cls, att_list in attention_by_label.items():
            mean_att = np.mean(att_list, axis=0)
            tokens = tokens_by_label[cls][0]

            non_special_tokens = [i for i, token in enumerate(tokens) if token not in tokenizer.all_special_tokens]
            truncated_attention = mean_att[non_special_tokens, :][:, non_special_tokens]
            truncated_tokens = [tokens[i] for i in non_special_tokens]

            plt.figure(figsize=(10, 8))
            sns.heatmap(truncated_attention, cmap='viridis',
                        xticklabels=truncated_tokens,
                        yticklabels=truncated_tokens,
                        vmin=0, vmax=truncated_attention.max() * 0.6)
            plt.xticks(rotation=90)
            plt.title(f'Mean Attention Heatmap - Class {cls}')
            plt.xlabel('Key Token Position')
            plt.ylabel('Query Token Position')
            plt.tight_layout()

            plt.savefig(f"images/mean_attention_map_class_{cls}.png")
            plt.close()

if __name__ == '__main__':
    evaluate_model()
