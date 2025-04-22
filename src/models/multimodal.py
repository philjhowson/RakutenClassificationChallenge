import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from CNN_custom_functions import resizing, EarlyStopping
from text_custom_functions import safe_loader, safe_saver
from multimodal_custom_functions import MultimodalData, multimodal_collate, MultimodalModel
import os
import json

def resize_image(img):
    return resizing(img, 224)

def train_multimodal_model(text = None, image = None):

    """
    sets the image folder and then loads the data and the indices for the
    training, validation, and test data.
    """

    img_dir = 'data/images/image_train/'

    data = pd.read_parquet('data/processed/translated_text.parquet')

    with open('data/processed/test_label_dictionary.json', 'r') as f:
        labels = json.load(f)

    labels = {int(key): value for key, value in labels.items()}
    data['labels'] = data['labels'].map(labels)

    train_indices = safe_loader('data/processed/train_indices.pkl')
    val_indices = safe_loader('data/processed/val_indices.pkl')
    test_indices = safe_loader('data/processed/test_indices.pkl')

    train = data.iloc[train_indices]
    val = data.iloc[val_indices]
    testing = data.iloc[test_indices]

    """
    if the text model is roBERTa-base, which is trained on English
    """

    if text == 'english':

        training = pd.read_csv('data/processed/train_text.csv')
        training['image_name'] = train['image_name'] + '.jpg'
        training = training[['image_name', 'designation_filtered', 'labels']]
        validation = pd.read_csv('data/processed/validation_text.csv')
        validation['image_name'] = val['image_name'] + '.jpg'
        validation = validation[['image_name', 'designation_filtered', 'labels']]        
        test = pd.read_csv('data/processed/test_text.csv')
        test['image_name'] = test['image_name'] + '.jpg'
        test = test[['image_name', 'designation_filtered', 'labels']]  

        training.to_csv('data/processed/train_english_multimodal.csv', index = False)
        validation.to_csv('data/processed/validation_english_multimodal.csv', index = False)
        test.to_csv('data/processed/test_english_multimodal.csv', index = False)

        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        text_model = RobertaForSequenceClassification.from_pretrained('roberta-base',  num_labels = 27,
                                                               attn_implementation = 'eager')

        for i, layer in enumerate(text_model.roberta.encoder.layer):
            if i < len(text_model.roberta.encoder.layer) - 1:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        text_model.load_state_dict(torch.load('models/roBERTa_model_weights.pth',
                                         weights_only = True))

        text = 'roBERTa'

    elif text == 'multi':

        training = pd.read_csv('data/processed/train_multilang.csv')
        training['image_name'] = train['image_name'] + '.jpg'
        training = training[['image_name', 'filtered_text', 'labels']]
        validation = pd.read_csv('data/processed/validation_multilang.csv')
        validation['image_name'] = val['image_name'] + '.jpg'
        validation = validation[['image_name', 'filtered_text', 'labels']]        
        test = pd.read_csv('data/processed/test_multilang.csv')
        test['image_name'] = test['image_name'] + '.jpg'
        test = test[['image_name', 'filtered_text', 'labels']]  

        training.to_csv('data/processed/train_multi_multimodal.csv', index = False)
        validation.to_csv('data/processed/validation_multi_multimodal.csv', index = False)
        test.to_csv('data/processed/test_multi_multimodal.csv', index = False)

        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        text_model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                               num_labels = 27,
                                                               attn_implementation = 'eager')

        for i, layer in enumerate(text_model.roberta.encoder.layer):
            if i < len(text_model.roberta.encoder.layer) - 1:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        text_model.load_state_dict(torch.load('models/roBERTa_multi_model_weights.pth',
                                         weights_only = True))

        text = 'roBERTa_multi'

        for i, layer in enumerate(roBERTa.roberta.encoder.layer):
            if i < len(roBERTa.roberta.encoder.layer) - 5:
                for param in layer.parameters():
                    param.requires_grad = False

    else:
        
        print("Incompatible --text argument. Please choose 'english' or 'multi'")
        return

    if image == 'densenet':

        image_model = models.densenet169(weights = None)
        image_model.classifier = nn.Linear(densenet.classifier.in_features, 27)
        image_model.load_state_dict(torch.load('models/densenet_model_weights.pth',
                                            weights_only = True))

        for param in image_model.parameters():
            param.requires_grad = False

        image_model = nn.Sequential(*list(image_model.children())[:-1],
                                 nn.AdaptiveAvgPool2d((1, 1)))

    elif image == 'resnet':

        image_model = models.resnet152(weights = None)
        image_model. fc = nn.Linear(model.fc.in_features, 27)
        image_model.load_state_dict(torch.load('models/resnet_model_weights.pth',
                                            weights_only = True))

        for param in image_model.parameters():
            param.requires_grad = False

        image_model = nn.Sequential(*list(image_model.children())[:-1],
                                 nn.AdaptiveAvgPool2d((1, 1)))

    else:
        
        print("Incompatible --image argument. Please choose 'densenet' or 'resnet'")
        return

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

    workers = os.cpu_count() // 2

    training = training.iloc[0:5]
    validation = validation.iloc[0:5]

    training_data = MultimodalData(training, img_dir = img_dir, transform = training_transforms,
                                   tokenizer = tokenizer)
    training_loader = DataLoader(training_data, batch_size = 5, shuffle = True,
                                 collate_fn = multimodal_collate,
                                 num_workers = workers, pin_memory = True)
    validation_data = MultimodalData(validation, img_dir = img_dir, transform = validation_transforms,
                                     tokenizer = tokenizer)
    validation_loader = DataLoader(validation_data, batch_size = 5, shuffle = False,
                                   collate_fn = multimodal_collate,
                                   num_workers = workers, pin_memory = True)


    """
    training_data = MultimodalData(training, img_dir = img_dir, transform = training_transforms,
                                   tokenizer = tokenizer)
    training_loader = DataLoader(training_data, batch_size = 64, shuffle = True,
                                 collate_fn = multimodal_collate,
                                 num_workers = workers, pin_memory = True)
    validation_data = MultimodalData(validation, img_dir = img_dir, transform = validation_transforms,
                                     tokenizer = tokenizer)
    validation_loader = DataLoader(validation_data, batch_size = 64, shuffle = False,
                                   collate_fn = multimodal_collate,
                                   num_workers = workers, pin_memory = True)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultimodalModel(text_model, image_model)
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

            with torch.autocast(device_type = 'cuda'):
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

    """

    model.load_state_dict(early_stopping.best_f1_model)
    torch.save(model.state_dict(), f'models/{text}+{image}_model_weights.pth')

    safe_saver(history, 'metrics/{text}+{image}_performance.pkl')

    print('Multimodal model and training metrics saved.')

    """
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train multimodal model.')
    parser.add_argument('--text', type = str, required = True, help = 'either english or multi')
    parser.add_argument('--img', type = str, required = True, help = 'either densenet or resnet')

    args = parser.parse_args()
    train_multimodal_model(args.text, args.cnn)

    
