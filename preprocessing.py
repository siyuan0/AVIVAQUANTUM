from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
with open('data/training_data.json', 'rb') as f:
    training_doc = json.load(f)

def text_preprocess(text_list, word_index):
    