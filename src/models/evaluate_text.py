import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from datasets import Dataset
from text_custom_functions import safe_loader, safe_saver, tokenize_function
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import argparse

def evaluate_model(text = None):

    """
    loads in the saved model based on user specifications and freezes all layers
    except the last layer so that gradcam can use the model gradients.
    """

    if text == 'english':

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels = 27,
                                                               attn_implementation = 'eager')

        for i, layer in enumerate(model.roberta.encoder.layer):
            if i < len(model.roberta.encoder.layer) - 1:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        model.load_state_dict(torch.load('models/roBERTa_model_weights.pth',
                                         weights_only = True))

        text = 'roBERTa'
        test = pd.read_csv('data/processed/test_text.csv').dropna()
        test.rename(columns = {'filtered_designation' : 'filtered_text'},
                    inplace = True)

    elif text == 'multi':

        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                               num_labels = 27,
                                                               attn_implementation = 'eager')

        for i, layer in enumerate(model.roberta.encoder.layer):
            if i < len(model.roberta.encoder.layer) - 1:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        model.load_state_dict(torch.load('models/roBERTa_multi_model_weights.pth',
                                         weights_only = True))

        text = 'roBERTa_multi'
        test = pd.read_csv('data/processed/test_multilang.csv')

    else:
        
        print("Incompatible --text argument. Please choose 'english' or 'multi'")
        return

    """
    loads in the test data, moves the model to GPU if it's available and sets
    the model into evaluation mode.
    """

    test_dataset = Dataset.from_pandas(test)

    test_dataset = test_dataset.map(lambda x : tokenize_function(x, column = 'filtered_text',
                                                                 tokenizer = tokenizer, length = 128),
                                    batched = True, num_proc = 4)
    test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    model.eval()

    """
    creates a list of labels and predicted labels and fills the list with each
    batch for the model evaluation. Saves the model classification report
    and model test f1-score.
    """

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
        
    safe_saver(test_report, f'metrics/{text}_classification_report.pkl')
    history = safe_loader(f'metrics/{text}_performance.pkl')

    labels = ['Training F1-Score', 'Validation F1-Score', 'Test F1-Score']
    scores = [max(history['f1']), max(history['val_f1']), test_f1]

    """
    creates a barplot of the f1-scores for traning, validation, and test
    scores and saves the plot.
    """

    fig = plt.figure(figsize = (20, 10))

    bars = plt.bar(labels, scores, color = ['blue', 'purple', 'green'])

    for bar in bars:
        yval = round(bar.get_height(), 3)
        plt.text(bar.get_x() + bar.get_width() /2, yval - 0.05, str(yval),
                 ha = 'center', va = 'bottom', color = 'white', fontweight = 'bold',
                 fontsize = 12)

    plt.ylabel('F1-Score')
    plt.title('Training, Validation, and Text F1-Scores')
    plt.tight_layout()

    plt.savefig(f'images/{text}_f1_scores.png')
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize = (20, 5))

    titles = ['Loss', 'F1-Score', 'Gradient']
    training_items = ['loss', 'f1', 'gradient']
    validation_items = ['val_loss', 'val_f1']

    """
    starts the ticks at one, and ends at the length of the history file (i.e.,
    the total number of epochs. Creates a performance history plot for
    training and validation metrics.
    """

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
    plt.savefig(f'images/{text}_training_history.png')
    plt.close()

    """
    opens the previously created test_label_dictionary during CNN evaluation
    and converts the labels to their original values for plotting.
    """

    with open('data/processed/test_label_dictionary.json', 'r') as f:
        label_dictionary = json.load(f)

    original_labels = {v: k for k, v in label_dictionary.items()}

    attention_by_label = defaultdict(list)
    tokens_by_label = defaultdict(list)

    """
    extracts attnetion maps for each of the tokens in the test loader and
    plots a heatmap of the amount of attention each gets during evaluation.
    """

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask = attention_mask, output_attentions = True)

            logits = outputs.logits
            preds = torch.argmax(logits, dim = 1)
            mapped_preds = [original_labels[pred.item()] for pred in preds]

            last_layer_attention = outputs.attentions[-1]
            attention_mean = last_layer_attention.mean(dim = 1)

            for i, pred_class in enumerate(mapped_preds):
                attention_by_label[pred_class].append(attention_mean[i].cpu().numpy())
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu())
                tokens = [token for token in tokens if token not in tokenizer.all_special_tokens]
                tokens_by_label[pred_class].append(tokens)


        for cls, att_list in attention_by_label.items():
            mean_att = np.mean(att_list, axis = 0)
            tokens = tokens_by_label[cls][0]

            non_special_tokens = [i for i, token in enumerate(tokens) if token not in tokenizer.all_special_tokens]
            truncated_attention = mean_att[non_special_tokens, :][:, non_special_tokens]
            truncated_tokens = [tokens[i] for i in non_special_tokens]

            plt.figure(figsize = (10, 8))
            sns.heatmap(truncated_attention, cmap = 'viridis',
                        xticklabels = truncated_tokens,
                        yticklabels = truncated_tokens,
                        vmin = 0, vmax = truncated_attention.max() * 0.6)
            plt.xticks(rotation = 90)
            plt.title(f'Mean Attention Heatmap - Class {cls}')
            plt.xlabel('Key Token Position')
            plt.ylabel('Query Token Position')
            plt.tight_layout()

            plt.savefig(f"images/mean_attention_map_{text}_class_{cls}.png")
            plt.close()

    print('Model evaluation complete. Metrics saved.')

if __name__ == '__main__':
    parser =  argparse.ArgumentParser(description = 'Evaluated text model.')
    parser.add_argument('--text', type = str, required = True, help = 'either english or multi')

    args = parser.parse_args()
    evaluate_model(args.text)
