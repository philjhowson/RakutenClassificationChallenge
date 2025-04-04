from sklearn.model_selection import train_test_split
import pandas as pd
import spacy
import pickle


def filter_dataset():

    with open('data/processed/test_indices.pkl', 'rb') as f:
        test_indices = pickle.load(f)

    data = pd.read_parquet('data/processed/translated_text.parquet')
    data = data[['designation_translation', 'prdtypecode']]

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

    test_data = data.iloc[test_indices]
    train_mask = ~data.index.isin(test_indices)
    data = data.iloc[train_mask]

    test_data.to_csv('data/processed/text_test_set.csv', index = False)

    training, validation = train_test_split(data, test_size = 0.3, random_state = 42)

    training.to_csv('data/processed/text_training_set.csv', index = False)
    validation.to_csv('data/processed/text_validation_set.csv', index = False)

if __name__ == "__main__":
    filter_dataset()
