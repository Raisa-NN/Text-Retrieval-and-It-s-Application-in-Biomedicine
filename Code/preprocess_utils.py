"""
Responsibility:
- Normalize text to improve lexical retrieval (BM25)
"""

import json
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def read_txt_files_from_directory(directory_path):
    file_contents = {}
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_contents[filename] = file.read()
                except Exception as e:
                    print(f"An error occurred while reading {filename}: {e}")
    except Exception as e:
        print(f"An error occurred while accessing the directory: {e}")
        return {}
    return file_contents

def load_from_json(filename):
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

def remove_stop_words(tokens):
    # NLTK stopwords requires the "stopwords" corpus to be downloaded.
    # On some machines/environments, it may not be present yet, causing a LookupError.
    # Use try/except so the pipeline still runs (with a small fallback list)
    # rather than crashing during preprocessing.
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = {"the", "and", "of", "to", "in", "a", "is", "for", "with", "on"}

    filtered_sentence = []
    for token in tokens:
        if token not in stop_words:
            filtered_sentence.append(token)
    return filtered_sentence

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def pre_process(text):
    text_lower = text.lower()
    tokens_no_punctuation = remove_punctuation(text_lower)
    filtered_tokens = remove_stop_words(tokens_no_punctuation)
    stemmed_tokens = stemming(filtered_tokens)
    return stemmed_tokens