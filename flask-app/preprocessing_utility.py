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


        