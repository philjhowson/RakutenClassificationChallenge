import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from CNN_custom_functions import resizing, EarlyStopping
from text_custom_functions import tokenize_function, safe_loader, safe_saver

def resize_image(img):
    return resizing(img, 224)

def train_multimodal_model():

    img_dir = 'data/images/image_train/'

    data = pd.read_parquet('data/processed/formatted_text.parquet')
    data['image_name'] = data['image_name'] + '.jpg'
    test_index = safe_loader('data/processed/test_indices.pkl')

    test = data.iloc[test_index]
    data = ~data.index.isin(test_index)

    training, validation = train_test_split(data, test_size = 0.3)


if __name__ == '__main__':
    train_multimodal_model()

    
