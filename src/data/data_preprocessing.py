# data preprocessing

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging
import re

# logging configuration
logger = logging.getLogger('data_transformation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
words_to_remove = ["not", "no", "never", "neither", "nor", "very", 
                   "really", "too", "extremely", "quite", "but", "however", 
                   "although", "though", "if", "unless", "except"]

stop_words = [word for word in stop_words if word not in words_to_remove]

def normalize_text(data):
    """Normalize the text data."""
    try:
        #1. Removing URLS
        data = re.sub('http\S+', '', data).strip()
        data = re.sub('www\S+', '', data).strip()

        #2. Removing Tags
        data = re.sub('#\S+', '', data).strip()

        #3. Removing Mentions
        data = re.sub('@\S+', '', data).strip()
        
        #4. Removing upper brackets to keep negative auxiliary verbs in text
        data = data.replace("'", "")    # ignoring ' as in don't 
        
        #5. Tokenize
        text_tokens = word_tokenize(data.lower())
        
        #6. Remove Puncs and number
        tokens_without_punc = [w for w in text_tokens if w.isalpha()]
        
        #7. Removing Stopwords
        tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
        
        #8. lemma
        text_cleaned = [WordNetLemmatizer().lemmatize(t) for t in tokens_without_sw]
        
        #joining
        return " ".join(text_cleaned)
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('data loaded properly')

        # Transform the data
        train_data["text"] = train_data["text"].apply(normalize_text) 
        test_data["text"] = test_data["text"].apply(normalize_text)
        
        train_processed_data = train_data
        test_processed_data = test_data

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()