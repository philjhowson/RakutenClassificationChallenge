import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.init as init
import torch
from PIL import Image

class MultimodalData(Dataset):
    def __init__(self, categories, img_dir, transform, tokenizer):
        self.images = categories.iloc[:, 0]
        self.text = categories.iloc[:, 1]
        self.labels = categories.iloc[:, 2]
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images.iloc[idx])
        label = torch.tensor(self.labels.iloc[idx])

        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)

        text = self.tokenizer(self.text.iloc[idx], return_tensors = "pt",
                              padding = 'max_length', truncation = True,
                              max_length = 128)

        return text, image, label
    
def multimodal_collate(batch):
    texts, images, labels = zip(*batch)

    images = torch.stack(images)
    labels = torch.stack(labels)

    input_ids = torch.stack([item['input_ids'].squeeze(0) for item in texts])
    attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in texts])

    text_batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    return text_batch, images, labels

class MultimodalModel(nn.Module):
    def __init__(self, txt_model, img_model):
        super().__init__()
        self.txt = txt_model
        self.img = nn.Sequential(*list(img_model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2432, 27)

        init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, images):
        outputs = self.txt(input_ids = input_ids,
                           attention_mask = attention_mask,
                           output_hidden_states = True)
        last_hidden_state = outputs.hidden_states[-1]
        cls_rep = last_hidden_state[:, 0, :]

        images = self.img(images)
        images = self.pool(images)
        images = images.view(images.size(0), -1)
    
        features = torch.cat([cls_rep, images], dim = 1)
        
        return self.classifier(features)
