import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
import json

def resize_image(img):
    return resizing(img, 224)

def extract_features():

    img_dir = 'data/images/image_train/'

    training = pd.read_csv('data/processed/train_multimodal.csv')
    validation = pd.read_csv('data/processed/validation_multimodal.csv')
    test = pd.read_csv('data/processed/test_multimodal.csv')

    transform = transforms.Compose([
        transforms.Lambda(resize_image),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    workers = os.cpu_count() // 2

    training_data = MultimodalData(training, img_dir = img_dir, transform = transform,
                                   tokenizer = tokenizer)
    training_loader = DataLoader(training_data, batch_size = 64, shuffle = True,
                                 collate_fn = multimodal_collate,
                                 num_workers = workers, pin_memory = True)
    validation_data = MultimodalData(validation, img_dir = img_dir, transform = transform,
                                     tokenizer = tokenizer)
    validation_loader = DataLoader(validation_data, batch_size = 64, shuffle = False,
                                   collate_fn = multimodal_collate,
                                   num_workers = workers, pin_memory = True)
    test_data = MultimodalData(test, img_dir = img_dir, transform = transform,
                               tokenizer = tokenizer)
    test_loader = DataLoader(test_data, batch_size = 64, shuffle = False,
                             collate_fn = multimodal_collate,
                             num_workers = workers, pin_memory = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    densenet = models.densenet169(weights = None)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, 27)

    for param in densenet.parameters():
        param.requires_grad = False

    densenet = nn.Sequential(*list(densenet.children())[:-1],
                             nn.AdaptiveAvgPool2d((1, 1)))

    roBERTa = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                               num_labels = 27)
    for param in roBERTa.parameters():
        param.requires_grad = False

    model = MultimodalModel(roBERTa, densenet, return_features = True)
    model = model.to(device)
    model.load_state_dict(torch.load('models/roBERTa_multi+densenet_model_weights.pth',
                                     weights_only = True))

    model = model.to(device)

    model.eval()
    training = []
    validation = []
    test = []

    for index, loader in enumerate([training_loader,
                                    validation_loader,
                                    test_loader]):

        with torch.no_grad():

            for batch in loader:

                text, images, labels = batch
                input_ids, attention_mask = text['input_ids'].to(device), text['attention_mask'].to(device)
                images = images.to(device)

                outputs = model(input_ids, attention_mask, images)
                numpy_feats = outputs.detach().cpu().numpy()
                df = pd.DataFrame(numpy_feats)
                df['labels'] = labels

                if index == 0:
                    training.append(df)
                elif index == 1:
                    validation.append(df)
                elif index == 2:
                    test.append(df)

    training = pd.concat(training, axis = 0, ignore_index = True)
    validation = pd.concat(validation, axis = 0, ignore_index = True)
    test = pd.concat(test, axis = 0, ignore_index = True)

    training.to_parquet('data/processed/training_features.parquet')
    validation.to_parquet('data/processed/validation_features.parquet')                
    test.to_parquet('data/processed/test_features.parquet')

    print('Features extracted and saved!')

if __name__ == '__main__':
    extract_features()
