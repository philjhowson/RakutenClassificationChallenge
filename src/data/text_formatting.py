from sklearn.model_selection import train_test_split
import pandas as pd
from text_functions import safe_loader
import pickle
import spacy

def filter_dataset():

    data = pd.read_parquet('data/processed/translated_text.parquet')

    data = data[['designation_lang', 'designation_translation', 'designation_filtered',
                 'description_filtered', 'prdtypecode', 'image_name']]
    data['designation_lang'] = data['designation_lang'].replace({'unknown' : 'fr'})
    data['description_filtered'] = data['description_filtered'].fillna('')
    data['combined_text'] = data.apply(lambda row: f"{row['designation_filtered']} {row['description_filtered']}", axis = 1)
    data.drop(columns = ['designation_filtered', 'description_filtered'], inplace = True)

    hexcode = {'\x93': '"', '\x94': '"', '\x97': '-', '\xad': '-',
               '\x9c': 'œ', '\x80': '€', '\x91': "'", '\x96': '-',
               '\x92': "'", '\x8c': 'Œ', '—': '-', '¡' : '', '`': "'",
               '¿': '?'}

    data['designation_translation'] = data['designation_translation'].replace(hexcode, regex = True)
    data['combined_text'] = data['combined_text'].replace(hexcode, regex = True)

 
    languages = {'en', 'de', 'it', 'nl', 'pl', 'sv', 'ro', 'fr', 'pt', 'es'}

    spacy_models = {
        'en': 'en_core_web_sm',  # English
        'de': 'de_core_news_sm',  # German
        'it': 'it_core_news_sm',  # Italian
        'nl': 'nl_core_news_sm',  # Dutch
        'pl': 'pl_core_news_sm',  # Polish
        'sv': 'sv_core_news_sm',  # Swedish
        'ro': 'ro_core_news_sm',  # Romanian
        'fr': 'fr_core_news_sm',  # French
        'pt': 'pt_core_news_sm',  # Portuguese
        'es': 'es_core_news_sm'   # Spanish
    }

    spacy_nlp = {lang: spacy.load(model) for lang, model in spacy_models.items()}

    def process_text(lang, text):
        try:
            nlp = spacy_nlp.get(lang)
            if not nlp:
                return text

            doc = nlp(text)

            processed_words = [
                token.lemma_.lower() for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]

            return ' '.join(processed_words)

        except Exception as e:
            print(f"Error processing text: {e}")
            return text

    data['filtered_text'] = data.apply(lambda x: process_text(x['designation_lang'], x['combined_text']), axis = 1)
    data['filtered_designation'] = data.apply(lambda x: process_text('en', x['designation_translation']), axis = 1)

    original_labels = data['prdtypecode'].tolist()
    unique_labels = sorted(set(original_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    data['labels'] = data['prdtypecode'].map(label_to_index)
    data.drop(columns = ['prdtypecode', 'designation_lang', 'designation_translation', 'combined_text'], inplace = True)
    data.to_parquet('data/processed/formatted_text.parquet')

    data.drop(columns = ['image_name'], inplace = True)
    simple_data = data.copy()
    data.drop(columns = ['filtered_designation'], inplace = True)

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
        df = df.drop_duplicates(subset = ['filtered_text'])
        splits[split] = df
        
    splits['train'].to_csv('data/processed/train_multilang.csv', index = False)
    splits['val'].to_csv('data/processed/validation_multilang.csv', index = False)
    splits['test'].to_csv('data/processed/test_multilang.csv', index = False)

    data = simple_data
    data.drop(columns = ['filtered_text'], inplace = True)

    train = data.iloc[train_indices]
    val = data.iloc[val_indices]
    test = data.iloc[test_indices]

    split_data = {'train' : train,
                  'val': val,
                  'test': test}

    splits = {}

    for split in split_data.keys():
        df = split_data[split]
        df = df.drop_duplicates(subset = ['filtered_designation'])
        splits[split] = df
        
    
    splits['train'].to_csv('data/processed/train_text.csv', index = False)
    splits['val'].to_csv('data/processed/validation_text.csv', index = False)
    splits['test'].to_csv('data/processed/test_text.csv', index = False)

if __name__ == "__main__":
    filter_dataset()
