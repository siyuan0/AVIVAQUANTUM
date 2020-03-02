
import json
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pattern.en import lemma

def data_temp():
    appos = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "I would",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will"
    }

    with open('data/training_data.json', 'rb') as f:
        training_data = json.load(f)

    with open('data/holdout_data.json', 'rb') as f:
        holdout_data = json.load(f)

    training_data_1 = []

    for petition in training_data:
        i = petition['abstract']['_value']
        j = petition['label']['_value']
        k = petition['numberOfSignatures']
        
        training_data_1.append({'Abstract': i, 'Label': j, 'Signatures': k})

    stop_words = stopwords.words('english')

    abstract_sequence = []
    label_sequence = []
    signature_sequence = []

    for petition in training_data_1:
        abstract_lemm = lemma(petition['Abstract'])
        label_lemm = lemma(petition['Label'])


        abstract_tokens = word_tokenize(abstract_lemm)
        label_tokens = word_tokenize(label_lemm)

        abstract_nostopwords = [word.lower() for word in abstract_tokens if word not in stop_words]
        abstract_nopunct = [word for word in abstract_nostopwords if word.isalpha()]
        abstract_final = [appos[word] if word in appos else word for word in abstract_nopunct]

        label_nostopwords = [word.lower() for word in label_tokens if word not in stop_words]
        label_nopunct = [word for word in label_nostopwords if word.isalpha()]
        label_final = [appos[word] if word in appos else word for word in label_nopunct]


        abstract_sequence.append(abstract_final)
        label_sequence.append(label_final)
        signature_sequence.append(petition['Signatures'])

    return abstract_sequence, label_sequence, signature_sequence




