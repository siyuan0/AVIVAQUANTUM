import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import math
import random
import csv  
import os
from tensorflow.keras.utils import Sequence
from .data_temp_loader import data_temp

def listshuffle(l):
    # shuffles a list and returns a new list (not in place and does not reference the old list)
    # probably quite memory consuming
    l_new = [e for e in l]
    random.shuffle(l_new)
    return l_new

def loadSampleData():
    '''load a sample test data'''
    with open(os.path.join(os.getcwd(),'data/sampleAbstract.csv')) as f:
        reader = csv.reader(f)
        abstract = list(reader)
        abstract = abstract[1:]
    with open(os.path.join(os.getcwd(),'data/sampleLabel.csv')) as f:
        reader = csv.reader(f)
        label= list(reader)
        label = label[1:]
    with open(os.path.join(os.getcwd(),'data/sampleSignature.csv')) as f:
        reader = csv.reader(f)
        signature = list(reader)
        signature = [int(e[0]) for e in signature[1:]]
    return abstract*10, label*10, signature*10


class TextSequence(Sequence):
    '''data generator, inputs need to be of tensor type'''
    def __init__(self, x_set, y_set, batch_size=32, shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = [i for i in range(self.x.shape[0])] # creates the index to reference the data

    def __len__(self):
        return math.ceil(self.x.shape[0]/ self.batch_size)

    def __getitem__(self, idx):
        indexes = self.index[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = tf.gather(self.x, indexes)
        batch_y = tf.gather(self.y, indexes)

        # batch_x = tf.transpose(tf.random.shuffle(tf.transpose(batch_x)))

        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.index)

class dataloader:
    def __init__(self, num_of_sets, batch_size=64):
        abstract_text, label_text, signature_sequence = data_temp()
        # abstract_text, label_text, signature_sequence = loadSampleData()

        data_all = [e for e in zip(abstract_text, label_text, signature_sequence)]
        random.shuffle(data_all)
        abstract_text = [abstract for abstract, label, signature in data_all]
        label_text = [label for abstract, label, signature in data_all]
        signature_sequence = [signature for abstract, label, signature in data_all]

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(abstract_text+label_text)
        self.word_index = self.tokenizer.word_index # creates the index to convert words to numbers
        
        self.batch_size = batch_size
        self.abstract_maxlen = 30
        self.label_maxlen = 10

        abstract_sequence = self.tokenizer.texts_to_sequences([' '.join(word for word in text) for text in abstract_text])
        abstract_sequence = pad_sequences(abstract_sequence, maxlen=self.abstract_maxlen, padding='post', truncating='post')
        label_sequence = self.tokenizer.texts_to_sequences([' '.join(word for word in text) for text in label_text])
        label_sequence = pad_sequences(label_sequence, maxlen=self.label_maxlen, padding='post', truncating='post')

        abstract_tensor = tf.convert_to_tensor(abstract_sequence, dtype=tf.float32)
        label_tensor = tf.convert_to_tensor(label_sequence, dtype=tf.float32)
        # signature_cat = [(0 if s<50 else 1, s) for s in signature_sequence]
        signature_cat = [0 if s<50 else 1 for s in signature_sequence]
        # signature_cat = [s for s in signature_sequence]
        
        self.x_data = tf.concat([abstract_tensor,label_tensor],1)
        self.y_data = tf.convert_to_tensor(signature_cat, dtype=tf.float32)

        self.num_of_sets = num_of_sets
        self.set_size = math.floor(self.x_data.shape[0]/self.num_of_sets)
        self._index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index < self.num_of_sets - 1:
            x_train = tf.concat([self.x_data[:self._index*self.set_size], self.x_data[(self._index+1)*self.set_size:]],axis=0)
            y_train = tf.concat([self.y_data[:self._index*self.set_size], self.y_data[(self._index+1)*self.set_size:]],axis=0)
            x_val = self.x_data[self._index*self.set_size:(self._index+1)*self.set_size]
            y_val = self.y_data[self._index*self.set_size:(self._index+1)*self.set_size]
            train_gen = TextSequence(x_train, y_train, batch_size=self.batch_size, shuffle=True)
            self._index +=1
            return train_gen, (x_train, y_train), (x_val, y_val)
        elif self._index == self.num_of_sets - 1:
            x_train = self.x_data[:self._index*self.set_size]
            y_train = self.y_data[:self._index*self.set_size]
            x_val = self.x_data[self._index*self.set_size:]
            y_val =  self.y_data[self._index*self.set_size:]
            train_gen = TextSequence(x_train, y_train, batch_size=self.batch_size, shuffle=True)
            
            return train_gen, (x_train, y_train), (x_val, y_val)
        else:
            raise StopIteration
