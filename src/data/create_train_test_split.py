import pandas as pd
from sklearn.model_selection import train_test_split
import json
from text_functions import safe_saver

def create_train_test_split():
    """
    simple function to extract the indices for the training, validation,
    and test sets to ensure no data leakage.
    """

    data = pd.read_csv('data/raw/y_train.csv', index_col = 0)

    temp, test = train_test_split(data, test_size = 0.1,
                                  stratify = data['prdtypecode'],
                                  random_state = 42)
    train, val = train_test_split(temp, test_size = 0.1,
                                  stratify = temp['prdtypecode'],
                                  random_state = 42)

    train_indices = list(train.index)
    val_indices = list(val.index)
    test_indices = list(test.index)
    
    safe_saver(train_indices, 'data/processed/train_indices.pkl')
    safe_saver(val_indices, 'data/processed/val_indices.pkl')
    safe_saver(test_indices, 'data/processed/test_indices.pkl')

if __name__ == '__main__':
    create_train_test_split()
