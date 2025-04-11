import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import RobertaForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from datasets import Dataset
from text_custom_functions import EarlyStopping, tokenize_function, safe_saver
import pandas as pd
import numpy as np

def train_roBERTa():

    training = pd.read_csv('data/processed/train_text.csv')
    validation = pd.read_csv('data/processed/validation_text.csv')

    train_dataset = Dataset.from_pandas(training)
    validation_dataset = Dataset.from_pandas(validation)

    train_dataset = train_dataset.map(tokenize_function, batched = True, num_proc = 4)
    validation_dataset = validation_dataset.map(tokenize_function, batched = True, num_proc = 4)
    train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
    validation_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
    training_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = 128, shuffle = False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RobertaForSequenceClassification.from_pretrained('roberta-base',  num_labels = 27)

    for i, layer in enumerate(model.roberta.encoder.layer):
        if i < len(model.roberta.encoder.layer) - 5:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-4, betas = (0.9, 0.999),
                           eps = 1e-8, weight_decay = 1e-4)
    early_stopping = EarlyStopping()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',
                                                     factor = 0.1, patience = 3)

    labels_for_weights = training['labels']
    unique_classes = np.unique(labels_for_weights)
    
    sklearn_weights = compute_class_weight(class_weight = 'balanced',
                                           classes = unique_classes,
                                           y = labels_for_weights)
    weights = torch.tensor(sklearn_weights, dtype = torch.float32, device = device)
    criterion = nn.CrossEntropyLoss(weight = weights)

    epochs = 100

    history = {'loss': [], 'f1': [], 'gradient': [],
               'val_loss': [], 'val_f1': []}

    print('roBERTa loaded and ready to begin training.')

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        all_labels = []
        all_preds = []
        i = 0

        for batch in training_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            i += 1
            if i == len(training_loader):
                batch_grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        batch_grad_norm += param.grad.norm(2).item() ** 2
                batch_grad_norm = batch_grad_norm ** 0.5
        
            predicted_classes = torch.argmax(logits, dim = 1).detach().cpu().numpy()
            labels = labels.cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(predicted_classes)

        epoch_loss = running_loss / len(training_loader)
        epoch_f1 = f1_score(all_labels, all_preds, average = 'weighted')

        history['loss'].append(epoch_loss)
        history['f1'].append(epoch_f1)
        history['gradient'].append(batch_grad_norm)

        model.eval()

        val_loss = 0.0
        val_labs = []
        val_preds = []

        with torch.no_grad():

            for batch in validation_loader:

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask = attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                val_loss += loss.item()

                predicted_classes = torch.argmax(logits, dim = 1)
                labs = labels.cpu().numpy()
                predicted_labels = predicted_classes.cpu().numpy()

                val_labs.extend(labs)
                val_preds.extend(predicted_labels)

            val_loss /= len(validation_loader)
            val_f1 = f1_score(val_labs, val_preds, average = 'weighted')

            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)

        print(f"Training Loss: {round(epoch_loss, 3)}, Training F1-Score: {round(epoch_f1, 3)}, "
              f"Gradient: {round(batch_grad_norm, 3)}, Validation Loss: {round(val_loss, 3)}, "
              f"Validation F1-Score: {round(val_f1, 3)}")

        if early_stopping(val_loss, val_f1, model):
            break

        scheduler.step(val_loss)

    model.load_state_dict(early_stopping.best_f1_model)
    torch.save(model.state_dict(), f'models/roBERTa_model_weights.pth')

    safe_saver(history, 'metrics/roBERTa_performance.pkl')

    print('roBERTa model and training metrics saved.')

if __name__ == "__main__":
    train_roBERTa()
