import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# def lemmatization(text):
#     """Lemmatize the text."""
#     lemmatizer = WordNetLemmatizer()
#     text = text.split()
#     text = [lemmatizer.lemmatize(word) for word in text]
#     return " ".join(text)

# def remove_stop_words(text):
#     """Remove stop words from the text."""
#     stop_words = set(stopwords.words("english"))
#     text = [word for word in str(text).split() if word not in stop_words]
#     return " ".join(text)

# def removing_numbers(text):
#     """Remove numbers from the text."""
#     text = ''.join([char for char in text if not char.isdigit()])
#     return text

# def lower_case(text):
#     """Convert text to lower case."""
#     text = text.split()
#     text = [word.lower() for word in text]
#     return " ".join(text)

# def removing_punctuations(text):
#     """Remove punctuations from the text."""
#     text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#     text = text.replace('؛', "")
#     text = re.sub('\s+', ' ', text).strip()
#     return text

# def removing_urls(text):
#     """Remove URLs from the text."""
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# def remove_small_sentences(df):
#     """Remove sentences with less than 3 words."""
#     for i in range(len(df)):
#         if len(df.text.iloc[i].split()) < 3:
#             df.text.iloc[i] = np.nan

# def normalize_text(text):
#     text = lower_case(text)
#     text = remove_stop_words(text)
#     text = removing_numbers(text)
#     text = removing_punctuations(text)
#     text = removing_urls(text)
#     text = lemmatization(text)

#     return text


stop_words = set(stopwords.words('english'))
words_to_remove = ["not", "no", "never", "neither", "nor", "very", 
                   "really", "too", "extremely", "quite", "but", "however", 
                   "although", "though", "if", "unless", "except"]

stop_words = [word for word in stop_words if word not in words_to_remove]

def normalize_text(data):
        
        """Normalize the text data."""
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


        