import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, categories, img_dir, transform):
        """
        Args:
            categories: a dataframe with file names and their target classes.
            It should be structured ['filenames', 'labels'].
            image_dir (str): Directory with all the images.
            transform (callable): transformations to be applied to images.
        """
        self.images = categories.iloc[:, 0]
        self.img_dir = img_dir
        self.transform = transform
        self.labels = categories.iloc[:, 1]

    def __len__(self):
        """
        returns the length of the input dataframe (required by dataloader)
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.
        """
        img_name = os.path.join(self.img_dir, self.images.iloc[idx]) 
        label = self.labels.iloc[idx]

        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)

        return image, label

def resizing(img, size):
    """
    args:
        img: expects an img.
        size: takes one int, and returns an image of size x size.
    """
    img_array = np.array(img)
    """
    this is what filters out the color specifications. I lowered it as much as a could for good results while still
    being sensitive to some light yellows and blues.
    """
    locations = np.where(np.all(img_array <= [248, 235, 235], axis = -1))
    """
    sometimes the image locations returns as a 1D array when the full image is an image, in that case
    resize to desired shape and save it.
    """
    if len(locations[0]) == 0 or len(locations[1]) == 0:
        pil_image = Image.fromarray(img_array)
        new_pil_image = pil_image.resize((size, size), Image.LANCZOS)
    
        return new_pil_image

    else:    
        coords = [np.min(locations[1]), np.max(locations[1]), np.min(locations[0]), np.max(locations[0])]
        image2 = img_array[coords[2]:coords[3], coords[0]:coords[1]]
        pil_image = Image.fromarray(image2)
        ratio = (pil_image.width * pil_image.height) / (500 * 500)
        new_width, new_height = size, size
        aspect_ratio = pil_image.width / pil_image.height
        """
        selects operations based on the largest size and then calculates the relative increase needed for the
        smaller size to maintain ratios while having the largest size at 300 pixels. White padding is then
        added on the top/bottom or left/right in equal preportions. Finally, the image is saved.
        """
        if pil_image.width > pil_image.height:
            new_height = int(new_width / aspect_ratio)
            resized_image_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            if resized_image_pil.height < size:
                value = size - resized_image_pil.height
                half2 = value // 2
                half1 = value - half2
                top = np.full((half1, size, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                bottom = np.full((half2, size, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                resized_image_np = np.array(resized_image_pil)
                new_image = np.concatenate((top, resized_image_np, bottom), axis = 0)
            else:
                new_image = np.array(resized_image_pil)
        
        else:
            new_width = int(new_height * aspect_ratio)
            resized_image_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            if resized_image_pil.width < size:
                value = size - resized_image_pil.width
                half2 = value // 2
                half1 = value - half2
                left = np.full((size, half1, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                right = np.full((size, half2, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                resized_image_np = np.array(resized_image_pil)
                new_image = np.concatenate((left, resized_image_np, right), axis = 1)
            else:
                new_image = np.array(resized_image_pil)

    """
    returns the image_name, ratio, aspect_ratio, coords, and xycoords for each plotting or analysis of the shapes of the images without the white borders.
    """
    return Image.fromarray(new_image)

class EarlyStopping:
    """
    args:
        patience: how many epochs should go by without loss decreasing
            before stopping is triggered. Default is 5.
        f1_patience: how many epochs should go by without f1 decreasing
            before stopping is triggered. Default is 10.
    """
    def __init__(self, patience = 5, f1_patience = 10):
        """
        initializes the patience, f1_patience, and variables to store
        best_loss, best_f1, counter, f1_counter, and the best_f1_model.
        """
        self.patience = patience
        self.f1_patience = f1_patience
        self.best_loss = float('inf')
        self.best_f1 = 0
        self.counter = 0
        self.f1_counter = 0
        self.best_f1_model = None

    def __call__(self, val_loss, f1_score, model):
        """
        the call which checks current loss and f1_score against
        current bests and either increases the counter(s) and/or
        saves the model and resets the relevant loss/f1_counter.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                print("Early stopping triggered due to Loss")

                return True

        if f1_score > self.best_f1:
            self.best_f1 = f1_score
            self.best_f1_model = model.state_dict()
            self.f1_counter = 0
        else:
            self.f1_counter += 1
            if self.f1_counter >= self.f1_patience:
                self.f1_counter = 0
                print("Early stopping triggered due to F1-Score")

                return True
                    
        return False
