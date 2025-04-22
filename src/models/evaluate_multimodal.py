import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from CNN_custom_functions import resizing
from text_custom_functions import safe_loader, safe_saver
from multimodal_custom_functions import MultimodalData, multimodal_collate, MultimodalModel
import json

def resize_image(img):
    return resizing(img, 224)

def evaluate_multimodal_model():

    img_dir = 'data/images/image_train/'

    test_transforms = transforms.Compose([
        transforms.Lambda(resize_image),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.486, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])

    if text == 'english':

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        text_model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                                 num_labels = 27)
        
        for param in text_model.parameters():
            param.requires_grad = False

        model.load_state_dict(torch.load('models/roBERTa_model_weights.pth',
                                         weights_only = True))
        text = 'roBERTa'
        data = pd.read_csv('data/processed/test_english_multimodal.csv')
        
    elif text == 'multi':

        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        text_model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base',
                                                               num_labels = 27)

        for param in text_model.parameters():
            param.requires_grad = False

        text_model.load_state_dict(torch.load('models/roBERTa_multi_model_weights.pth',
                                              weights_only = True))
        text = 'roBERTa_multi'
        data = pd.read_csv('data/processed/test_multi_multimodal.csv')

    else:
        print("Incompatible --text argument. Please choose 'english' or 'multi'")

    test_data = MultimodalData(data, img_dir = img_dir, transform = test_transforms,
                               tokenizer = tokenizer)
    test_loader = DataLoader(test_data, batch_size = 256, shuffle = False,
                             collate_fn = multimodal_collate, num_workers = 4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if image == 'densenet':
    
        image_model = models.densenet169(weights = None)
        image_model.classifier = nn.Linear(densenet.classifier.in_features, 27)

        for param in densenet.parameters():
            param.requires_grad = False

        image_model = nn.Sequential(*list(densenet.children())[:-1],
                                    nn.AdaptiveAvgPool2d((1, 1)))

    elif image == 'resnet':

        image_model = models.resnet152(weights = None)
        image_model.fc = nn.Linear(model.fc.in_features, 27)
        image_model.load_state_dict(torch.load('models/resnet_model_weights.pth',
                                               weights_only = True))
        
        for param in model.parameters():
            param.requires_grad = False

        image_model = nn.Sequential(*list(image_model.children())[:-1],
                                 nn.AdaptiveAvgPool2d((1, 1)))
        
    model = MultimodalModel(text_model, image_model)
    model = model.to(device)
    model.load_state_dict(torch.load(f'models/{text}+{image}_model_weights.pth',
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

    safe_saver(class_report, f'metrics/{text}+{image}_classification_report.pkl')
    multimodal_history = safe_loader(f'metrics/{text}+{image}+densenet_performance.pkl')

    train_f1 = max(multimodal_history['f1'])
    validation_f1 = max(multimodal_history['val_f1'])

    f1_labels = ['Training F1-Score', 'Validation F1-Score', 'Test F1-Score']
    f1_values = [train_f1, validation_f1, f1]

    fig = plt.figure(figsize = (15, 5))

    bars = plt.bar(f1_labels, f1_values, color = ['blue', 'purple', 'green'])

    for bar in bars:
        yval = round(bar.get_height(), 3)
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.1, str(yval),
                 ha = 'center', va = 'bottom', color = 'white',
                 fontweight = 'bold')

    plt.ylim(0, 1)
    plt.ylabel('F1-Score')
    plt.title('Training, Validation, and Test F1-Scores')
    plt.tight_layout()

    plt.savefig(f'images/{text}+{image}_f1_scores.png')

    print('Multimodal model evaluation metrics saved.')

if __name__ == '__main__':
    evaluate_multimodal_model()
