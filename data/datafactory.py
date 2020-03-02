import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import math
import random
from tensorflow.keras.utils import Sequence
from .data_temp_loader import data_temp

def listshuffle(l):
    # shuffles a list and returns a new list (not in place and does not reference the old list)
    # probably quite memory consuming
    l_new = [e for e in l]
    random.shuffle(l_new)
    return l_new


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

def dataloader():
    abstract_text, label_text, signature_sequence = data_temp()

    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(abstract_text+label_text)
    word_index = tokenizer.word_index # creates the index to convert words to numbers

    abstract_maxlen = 50
    label_maxlen = 15

    abstract_sequence = tokenizer.texts_to_sequences([' '.join(word for word in text) for text in abstract_text])
    abstract_sequence = pad_sequences(abstract_sequence, maxlen=abstract_maxlen, padding='post', truncating='post')
    label_sequence = tokenizer.texts_to_sequences([' '.join(word for word in text) for text in label_text])
    label_sequence = pad_sequences(label_sequence, maxlen=label_maxlen, padding='post', truncating='post')

    abstract_tensor = tf.convert_to_tensor(abstract_sequence, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(label_sequence, dtype=tf.float32)
    signature_cat = [0 if s<50 else 1 for s in signature_sequence]

    x_data = tf.concat([abstract_tensor,label_tensor],1)
    y_data = tf.convert_to_tensor(signature_cat, dtype=tf.float32)
    
    x_train = x_data[:10000]
    y_train = y_data[:10000]
    x_val = x_data[10000:]
    y_val = y_data[10000:]

    train_gen = TextSequence(x_train, y_train)

    return train_gen, (x_val, y_val)