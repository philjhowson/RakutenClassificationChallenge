import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import pickle
from CNN_custom_functions import ImageDataset, resizing, EarlyStopping

def resize_image(img):
    return resizing(img, 224)

def train_densenet():

    train = pd.read_csv('data/processed/train_CNN.csv')
    val = pd.read_csv('data/processed/validation_CNN.csv')

    img_dir = 'data/images/image_train/'

    unique_labels = sorted(set(train['target']))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    train['target'] = train['target'].map(label_to_index)
    val['target'] = val['target'].map(label_to_index)

    train_transform = transforms.Compose([
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

    test_transform = transforms.Compose([
        transforms.Lambda(resize_image),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])

    train_data = ImageDataset(categories = train, img_dir = img_dir,
                              transform = train_transform)
    val_data = ImageDataset(categories = val, img_dir = img_dir,
                            transform = test_transform)

    train_loader = DataLoader(train_data, batch_size = 64, shuffle = True,
                              num_workers = 4)
    val_loader = DataLoader(val_data, batch_size = 64, shuffle = False,
                            num_workers = 4)

    model = models.densenet169(weights = 'DenseNet169_Weights.DEFAULT')
    model.classifier = nn.Linear(model.classifier.in_features, 27)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
            
    layers = [model.features.denseblock4.denselayer28,
              model.features.denseblock4.denselayer29,
              model.features.denseblock4.denselayer30,
              model.features.denseblock4.denselayer31,
              model.features.denseblock4.denselayer32]

    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr = 1e-4, betas = (0.9, 0.999),
                           eps = 1e-8, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',
                                                     factor = 0.1, patience = 3)
    early_stopping = EarlyStopping(patience = 6)

    labels = train['target'].tolist()
    sklearn_weights = compute_class_weight(class_weight = 'balanced',
                                           classes = np.unique(labels), y = labels)
    weights = torch.tensor(sklearn_weights, dtype = torch.float32, device = device)
    criterion = nn.CrossEntropyLoss(weight = weights)

    epochs = 100
    history = {'loss' : [], 'f1' : [], 'gradient' : [],
               'val_loss' : [], 'val_f1' : []}

    print('DenseNet169 loaded and ready to begin training.')

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        all_labels = []
        all_preds = []
        i = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            i += 1
            if i == len(train_loader):
                batch_grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        batch_grad_norm += param.grad.norm(2).item() ** 2
                batch_grad_norm = batch_grad_norm ** 0.5

            predicted_classes = torch.argmax(outputs, dim = 1)
            labs = labels.cpu().numpy()
            preds = predicted_classes.cpu().numpy()

            optimizer.step()
            running_loss += loss.item()
            all_labels.extend(labs)
            all_preds.extend(preds)

        epoch_loss = running_loss / len(train_loader)
        epoch_f1 = f1_score(all_labels, all_preds, average = 'weighted')

        history['loss'].append(epoch_loss)
        history['f1'].append(epoch_f1)
        history['gradient'].append(batch_grad_norm)

        model.eval()
        val_loss = 0.0
        val_labs = []
        val_preds = []

        with torch.no_grad():

            for val_images, val_labels in val_loader:
                
                val_images, val_labels = val_images.to(device), val_labels.to(device).long()
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()

                predicted_classes = torch.argmax(val_outputs, dim = 1)
                val_labs.extend(val_labels.cpu().numpy())
                val_preds.extend(predicted_classes.cpu().numpy())

            val_f1 = f1_score(val_labs, val_preds, average = 'weighted')
            val_loss /= len(val_loader)

        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss {epoch_loss:.4f}, Gradient: "
              f"{batch_grad_norm:.4f}, F1-Score: {epoch_f1:.4f}, Validation Loss: "
              f"{val_loss:.4f}, Validation F1-Score: {val_f1:.4f}")

        if early_stopping(val_loss, val_f1, model):
            break

        scheduler.step(val_loss)

    model.load_state_dict(early_stopping.best_f1_model)
    torch.save(model.state_dict(), f'models/densenet_model_weights.pth')

    with open(f'metrics/densenet_performance.pkl', 'wb') as f:
        pickle.dump(history, f)

    print('DenseNet169 model and training metrics saved.')

if __name__ == "__main__":
    train_densenet()
