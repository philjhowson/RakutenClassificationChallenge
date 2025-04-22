import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.init as init
import torch
from PIL import Image

class MultimodalData(Dataset):
    def __init__(self, categories, img_dir, transform, tokenizer):
        """
        args:
            categories: takes a df that should be in the order: image name,
            text, labels.
            img_dir: the image directory
            transform: a transforms.Compose() object
            tokenizer: a tokenizer object
        """
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
    """
    this function is for collating the text and image data for use in
    the dataloader object.
    """
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
    def __init__(self, txt_model, img_model, return_features = False):
        """
        args:
            text_model: the text model to be used. For roBERTa style
                transformers, the attn_implementation argument should
                be set to 'eager', so that feature maps can be extracted.
            img_model: the image model to be used. Should have the
                classifier removed.
            return_features: an optional argument if the user wishes to
                return the fused features and not the classifications.
        """
        super().__init__()
        self.txt = txt_model
        self.img = img_model
        self.classifier = nn.Sequential(
            nn.Linear(2432, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 27)
        )
        self.return_features = return_features
        """
        initializes the weights using xavier initialization.
        """
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, images):
        """
        extracts the feature maps for text and image, then
        concatinates them and either returns the features or
        the classification output.
        """
        outputs = self.txt(input_ids = input_ids,
                           attention_mask = attention_mask,
                           output_hidden_states = True)
        last_hidden_state = outputs.hidden_states[-1]
        text = last_hidden_state[:, 0, :]

        images = self.img(images)
        images = images.view(images.size(0), -1)
    
        features = torch.cat([text, images], dim = 1)
        
        if self.return_features == True:
            return features
        
        return self.classifier(features)
