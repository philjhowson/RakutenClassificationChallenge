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
