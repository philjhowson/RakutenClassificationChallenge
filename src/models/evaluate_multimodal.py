import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CNN_custom_functions import resizing
from text_custom_functions import safe_loader, safe_saver
from multimodal_custom_functions import MultimodalData, multimodal_collate, MultimodalModel
import json

def resize_image(img):
    return resizing(img, 224)

def evaluate_multimodal_model():

    img_dir = 'data/images/image_train/'

    data = pd.read_parquet('data/processed/formatted_text.parquet')
    data['image_name'] = data['image_name'] + '.jpg'
    test_index = safe_loader('data/processed/test_indices.pkl')

    data = data.iloc[test_index]

    test_transforms = transforms.Compose([
        transforms.Lambda(resize_image),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.486, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    test_data = MultimodalData(data, img_dir = img_dir, transform = test_transforms,
                               tokenizer = tokenizer)
    test_loader = DataLoader(test_data, batch_size = 256, shuffle = False,
                             collate_fn = multimodal_collate, num_workers = 4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = models.resnet152(weights = None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 27)
    resnet.load_state_dict(torch.load('models/resnet_model_weights.pth',
                                        weights_only = True))

    for param in resnet.parameters():
        param.requires_grad = False

    resnet = nn.Sequential(*list(resnet.children())[:-1])

    roBERTa = RobertaForSequenceClassification.from_pretrained('roberta-base',  num_labels = 27)
    roBERTa.load_state_dict(torch.load('models/roBERTa_model_weights.pth', weights_only = True))

    for param in roBERTa.parameters():
        param.requires_grad = False

    model = MultimodalModel(roBERTa, resnet)
    model = model.to(device)
    model.load_state_dict(torch.load('models/roBERTa+resnet_model_weights.pth',
                                     weights_only = True))
    model.to(device)

    model.eval()

    all_preds = []
    all_labs = []
    
    for batch in test_loader:
        text, images, labels = batch
        ids, mask = text['input_ids'].to(device), text['attention_mask'].to(device)
        images = images.to(device)

        with torch.no_grad():

            outputs = model(ids, mask, images)
            
        predicted_classes = torch.argmax(outputs, dim = 1)
        preds = predicted_classes.cpu().detach().numpy()
        labs = labels.cpu().numpy()

        all_preds.extend(preds)
        all_labs.extend(labels)

    f1 = f1_score(all_labs, all_preds, average = 'weighted')
    class_report = classification_report(all_labs, all_preds, output_dict = True)

    safe_saver(class_report, 'metrics/roBERTa+resnet_classification_report.pkl')
    multimodal_history = safe_loader('metrics/roBERTa+resnet_performance.pkl')

    train_f1 = max(multimodal_history['f1'])
    validation_f1 = max(multimodal_history['val_f1'])

    f1_labels = ['Training F1-Score', 'Validation F1-Score', 'Test F1-Score']
    f1_values = [train_f1, validation_f1, f1]

    fig = plt.figure(figsize = (15, 5))

    bars = plt.bar(f1_labels, f1_values, color = ['blue', 'purple', 'green'])

    for bar in bars:
        yval = round(bar.get_height())
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.05, str(yval),
                 ha = 'center', va = 'bottom', color = 'white',
                 fontweight = 'bold')

    plt.ylim(0, 1)
    plt.ylabel('F1-Score')
    plt.title('Training, Validation, and Test F1-Scores')
    plt.tight_layout()

    plt.savefig('images/roBERa+resnet_f1_scores.png')

    densenet_history = safe_loader('metrics/densenet_classification_report.pkl')
    resnet_history = safe_loader('metrics/resnet_classification_report.pkl')
    roBERTa_history = safe_loader('metrics/roBERTa_classification_report.pkl')
    roBERTa_densenet_history = safe_loader('metrics/roBERTa+densenet_classification_report.pkl')

    densenet_f1 = densenet_history['weighted avg']['f1-score']
    resnet_f1 = resnet_history['weighted avg']['f1-score']
    roBERTa_f1 = roBERTa_history['weighted avg']['f1-score']
    roBERTa_densenet_f1 = roBERTa_densenet_history['weighted avg']['f1-score']

    f1_values = [densenet_f1, resnet_f1, roBERTa_f1, roBERTa_densenet_f1, f1]
    f1_labels = ['DenseNet F1-Score', 'ResNet F1-Score', 'roBERTa F1-Score',
                 'roBERTa + DenseNet F1-Score', 'roBERTa + ResNet F1-Score']

    plt.figure(figsize = (10, 5))

    bars = plt.bar(f1_labels, f1_values, color = ['blue', 'purple', 'black', 'orange', 'green'])

    for bar in bars:
        yval = round(bar.get_height(), 3)
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.05, str(yval),
                 ha = 'center', va = 'bottom', color = 'white',
                 fontweight = 'bold')

    plt.xticks(rotation = 45, ha = 'right', va = 'top')
    plt.ylim(0, 1)
    plt.ylabel('F1-Score')
    plt.title('Test Scores for DenseNet, ResNet, roBERTa, and the Mulitmodal Models')
    plt.tight_layout()

    plt.savefig('images/model_f1_comparisons.png')

    print('Multimodal model evaluation metrics saved.')

if __name__ == '__main__':
    evaluate_multimodal_model()
