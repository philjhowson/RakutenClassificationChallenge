from sklearn.model_selection import train_test_split
import pandas as pd
from text_functions import safe_loader
import spacy
import pickle

def filter_dataset():

    data = pd.read_parquet('data/processed/translated_text.parquet')
    data = data[['designation_translation', 'prdtypecode', 'image_name']]

    spacy.require_gpu()

    nlp = spacy.load('en_core_web_trf')

    data['designation_filtered'] = list(nlp.pipe(data['designation_translation']))
    data['designation_filtered'] = data['designation_filtered'].astype(str)
    data.drop(columns = ['designation_translation'], inplace = True)

    original_labels = data['prdtypecode'].tolist()
    unique_labels = sorted(set(original_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    data['labels'] = data['prdtypecode'].map(label_to_index)
    data.drop(columns = ['prdtypecode'], inplace = True)

    hexcode = {'\x93': '"', '\x94': '"', '\x97': '-', '\xad': '-',
               '\x9c': 'œ', '\x80': '€', '\x91': "'", '\x96': '-',
               '\x92': "'", '\x8c': 'Œ', '—': '-', '¡' : '', '`': "'",
               '¿': '?'}

    data['designation_filtered'] = data['designation_filtered'].replace(hexcode, regex = True)
    data.to_parquet('data/processed/formatted_text.parquet')
    data.drop(columns = ['image_name'], inplace = True)

    train_indices = safe_loader('data/processed/train_indices.pkl')
    val_indices = safe_loader('data/processed/val_indices.pkl')
    test_indices = safe_loader('data/processed/test_indices.pkl')

    train = data.iloc[train_indices]
    val = data.iloc[val_indices]
    test = data.iloc[test_indices]

    split_data = {'train' : train,
                  'val': val,
                  'test': test}

    splits = {}

    for split in split_data.keys():
        df = split_data[split]
        df = df.drop_duplicates(subset = ['designation_filtered'])
        splits[split] = df
        
    
    splits['train'].to_csv('data/processed/train_text.csv', index = False)
    splits['val'].to_csv('data/processed/validation_text.csv', index = False)
    splits['test'].to_csv('data/processed/test_text.csv', index = False)

if __name__ == "__main__":
    filter_dataset()
