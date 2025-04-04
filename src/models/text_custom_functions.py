from transformers import RobertaTokenizer
import pickle
import os

class EarlyStopping:
    def __init__(self, patience = 6):
        self.patience = patience
        self.best_loss = float('inf')
        self.best_f1 = 0
        self.counter = 0
        self.best_model = None
    def __call__(self, val_loss, val_f1, model):
        if val_f1 > self.best_f1:
            self.best_f1_model = model.state_dict()
            self.best_f1 = val_f1
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                print("Early stopping triggered")
                model.load_state_dict(self.best_f1_model)
                        
                return True
                    
        return False

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(dataset):
    return tokenizer(dataset['designation_filtered'], padding = 'max_length',
                     truncation = True, max_length = 128)

def safe_loader(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at: {path}")
    if os.path.getsize(path) == 0:
        raise EOFError(f"File is empty at: {path}")

    with open(path, 'rb') as f:
        item = pickle.load(f)

    return item

def safe_saver(item, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, 'wb') as f:
        pickle.dump(item, f)
