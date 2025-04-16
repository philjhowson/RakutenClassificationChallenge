import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from CNN_custom_functions import resizing, EarlyStopping
from text_custom_functions import safe_loader, safe_saver
from multimodal_custom_functions import MultimodalData, multimodal_collate, MultimodalModel
import os

def resize_image(img):
    return resizing(img, 224)

def train_multimodal_model():

    img_dir = 'data/images/image_train/'

    data = pd.read_parquet('data/processed/formatted_text.parquet')
    data['image_name'] = data['image_name'] + '.jpg'
    data = data.drop(columns = ['filtered_text'], inplace = True)
    train_indices = safe_loader('data/processed/train_indices.pkl')
    val_indices = safe_loader('data/processed/val_indices.pkl')
    test_indices = safe_loader('data/processed/test_indices.pkl')

    training = data.iloc[train_indices]
    validation = data.iloc[val_indices]
    test = data.iloc[test_indices]

    training.to_csv('data/processed/train_multimodal.csv', index = False)
    validation.to_csv('data/processed/validation_multimodal.csv', index = False)
    test.to_csv('data/processed/test_multimodal.csv', index = False)

    training, validation = train_test_split(validation, test_size = 0.2,
                                            random_state = 42)

    labels = training['labels']

    training_transforms = transforms.Compose([
        transforms.Lambda(resize_image),
        transforms.RandomVerticalFlip(p = 0.4),
        transforms.RandomHorizontalFlip(p = 0.4),
        transforms.RandomAffine(degrees = (-90, 90),
                                translate = (0.1, 0.1),
                                fill = [255, 255, 255]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])

    validation_transforms = transforms.Compose([
        transforms.Lambda(resize_image),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    workers = os.cpu_count() // 2

    training_data = MultimodalData(training, img_dir = img_dir, transform = training_transforms,
                                   tokenizer = tokenizer)
    training_loader = DataLoader(training_data, batch_size = 128, shuffle = True,
                                 collate_fn = multimodal_collate,
                                 num_workers = workers, pin_memory = False)
    validation_data = MultimodalData(validation, img_dir = img_dir, transform = validation_transforms,
                                     tokenizer = tokenizer)
    validation_loader = DataLoader(validation_data, batch_size = 128, shuffle = False,
                                   collate_fn = multimodal_collate,
                                   num_workers = workers, pin_memory = False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet = models.resnet152(weights = None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 27)
    resnet.load_state_dict(torch.load('models/resnet_model_weights.pth',
                                        weights_only = True))

    for param in resnet.parameters():
        param.requires_grad = False
            
    for param in resnet.layer4.parameters():
        param.requires_grad = True

    resnet = nn.Sequential(*list(resnet.children())[:-1])

    roBERTa = RobertaForSequenceClassification.from_pretrained('roberta-base',  num_labels = 27)
    roBERTa.load_state_dict(torch.load('models/roBERTa_model_weights.pth', weights_only = True))

    for i, layer in enumerate(roBERTa.roberta.encoder.layer):
        if i < len(roBERTa.roberta.encoder.layer) - 5:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    model = MultimodalModel(roBERTa, resnet)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-4, betas = (0.9, 0.999),
                           eps = 1e-8, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',
                                                     factor = 0.1, patience = 3)
    early_stopping = EarlyStopping(patience = 6)
    sklearn_weights = compute_class_weight(class_weight = 'balanced',
                                           classes = np.unique(labels), y = labels)
    weights = torch.tensor(sklearn_weights, dtype = torch.float32, device = device)
    criterion = nn.CrossEntropyLoss(weight = weights)

    history = {'loss': [], 'f1': [], 'gradient': [], 'val_loss': [], 'val_f1': []}
    
    epochs = 20

    for epoch in range(epochs):

        running_loss = 0.0
        i = 0
        all_preds = []
        all_labs = []

        for batch in training_loader:

            text, images, labels = batch
            input_ids, attention_mask = text['input_ids'].to(device), text['attention_mask'].to(device)
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()

            i += 1
            if i == len(training_loader):
                batch_grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        batch_grad_norm += param.grad.norm(2).item() ** 2
                batch_grad_norm = batch_grad_norm ** 0.5

            running_loss += loss.item()
            predicted_classes = torch.argmax(outputs, dim = 1)
            preds = predicted_classes.cpu().detach().numpy()
            labs = labels.cpu().numpy()

            optimizer.step()

            all_preds.extend(preds)
            all_labs.extend(labs)

        epoch_loss = running_loss / len(training_loader)
        epoch_f1 = f1_score(all_labs, all_preds, average = 'weighted')

        history['loss'].append(epoch_loss)
        history['f1'].append(epoch_f1)
        history['gradient'].append(batch_grad_norm)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labs = []

        with torch.no_grad():

            for batch in validation_loader:

                text, images, labels = batch
                input_ids, attention_mask = text['input_ids'].to(device), text['attention_mask'].to(device)
                images, labels = images.to(device), labels.to(device).long()

                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                labs = labels.cpu().numpy()

                predicted_classes = torch.argmax(outputs, dim = 1)
                preds = predicted_classes.cpu().numpy()
                val_labs.extend(labs)
                val_preds.extend(preds)

            val_loss /= len(validation_loader)
            val_f1 = f1_score(val_labs, val_preds, average = 'weighted')

        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss {epoch_loss:.4f}, Gradient: "
              f"{batch_grad_norm:.4f}, F1-Score: {epoch_f1:.4f}, Validation Loss: "
              f"{val_loss:.4f}, Validation F1-Score: {val_f1:.4f}")

        if early_stopping(val_loss, val_f1, model):
            break

        scheduler.step(val_loss)

    model.load_state_dict(early_stopping.best_f1_model)
    torch.save(model.state_dict(), f'models/roBERTa+resnet_model_weights.pth')

    safe_saver(history, 'metrics/roBERTa+resnet_performance.pkl')

    print('Multimodal model and training metrics saved.')

if __name__ == '__main__':
    train_multimodal_model()

    
