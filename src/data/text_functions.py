import ftfy
import unicodedata
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
import argostranslate.translate

def filter_nofilter( text ):
    return text

# filter that does nothing, it is needed for the looping
def filter_designation( text ):
    return

# filter that does nothing, it is needed for the looping
def filter_description( text ):
    return

# use ftfy to remove/replace wrong character set encodings
def filter_ftfy( text ):
    fixed_text = ftfy.fix_text( text )
    return fixed_text

# use unicodedata to remove/replace no printing characters
def filter_unicodedata( row ):
    fixed_text = ''.join( char for char in row if unicodedata.category(char) not in ["Cf", "Cc", "Zl", "Zp"] )
    return fixed_text

def filter_bs4( text ):
    return BeautifulSoup( text, features = 'html.parser' ).get_text( ' ' )

# remove usernames from the end of the string, pattern is: @'only no space/tab characters'EOL
def filter_username( text ):
    return re.sub(r"@[^\s]*$", "", text )

# find multiple space/tab and replace it with just a single space
def filter_whitespace( text ):
    re_combined_space = re.compile(r"\s+")
    cleaned_text = re_combined_space.sub(" ", text).strip()
    return cleaned_text

def filter_norm(text):
    return unicodedata.normalize('NFKC', text)

def filter_charnorm(text):
    # Step 1: Detect encoding with charset-normalizer
    byte_data = str.encode( text )
    detected = from_bytes(byte_data).best()
    if detected:
        try:
            # Step 2: Decode using the detected encoding
            #print( 'detect', detected )
            decoded_text = str( detected )
            return decoded_text
        except Exception as e:
            print(f"Decoding or fixing failed: {e}")
            return None
    else:
        print("Encoding could not be detected.")
        return None

def str_length( text ):
    return len(text)
    
# calculate the word count of string (word are separated by ' ' space)
def str_wordcount( text ):
    return len( text.split() )

def detect_encoding( text ):
    byte_data = str.encode( text )
    result = chardet.detect(byte_data)
    return result['encoding']

def convert_to_utf8(text, encoding=None):
    """Convert a byte-encoded text to a UTF-8 string using a specified or detected encoding."""
    byte_data = str.encode( text )
    try:
        # Step 1: Use the specified encoding if provided
        if encoding:
            decoded_text = byte_data.decode(encoding)
        else:
            # Step 2: If encoding is unknown, use chardet to detect it
            detected = chardet.detect(byte_data)
            detected_encoding = detected.get('encoding')
            decoded_text = byte_data.decode(detected_encoding) if detected_encoding else None
        # Step 3: Re-encode as UTF-8
        return decoded_text.encode('utf-8').decode('utf-8') if decoded_text else None
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        print(f"Error converting text: {byte_data} with encoding {encoding or detected_encoding}: {e}")
        return None

def base_name_func(function):
    return function.__name__

def filter_stacked(text, filter_func_list):
    filtered = text
    for filter_func in filter_func_list:
        filtered = filter_func( filtered )
    return filtered


def detect_and_translate_offline(df):
    DetectorFactory.seed = 0  # Ensure consistent language detection

    # Supported languages in langdetect (code mapping)
    SUPPORTED_LANGUAGES = {
        "fr": "fr", "en": "en", "de": "de", "es": "es",
        "it": "it", "sv": "sv", "pl": "pl", "nl": "nl",
        "ro": "ro", "pt": "pt", "ja": "ja"
    }
    
    def detect_language(text):
        """Detect language of given text, handling NaN cases."""
        if pd.isna(text) or text.strip() == "":
            return None  # Return None for NaN or empty strings
        try:
            lang = detect(text)
            return lang if lang in SUPPORTED_LANGUAGES else "unknown"
        except:
            return "unknown"  # Handle detection failures
    
    def translate_to_english_offline(text, source_lang):
        """Translate text to English using offline translation, handling NaN cases."""
        if pd.isna(text) or text.strip() == "" or source_lang in [None, "unknown", "en"]:
            return text  # Return original if already English or undetectable
        try:
            return argostranslate.translate.translate(text, source_lang, "en")
        except:
            return text  # Return original text if translation fails

    # Detect languages
    df["designation_lang"] = df["designation_filtered"].apply(detect_language)
    df["description_lang"] = df["description_filtered"].apply(detect_language)
    
    # Translate using offline translator
    df["designation_translation"] = df.apply(lambda row: translate_to_english_offline(row["designation_filtered"], row["designation_lang"]), axis=1)
    df["description_translation"] = df.apply(lambda row: translate_to_english_offline(row["description_filtered"], row["description_lang"]), axis=1)

    # Detect languages after translation to confirm it's English
    df["designation_lang_after"] = df["designation_translation"].apply(detect_language)
    df["description_lang_after"] = df["description_translation"].apply(detect_language)

    return df
