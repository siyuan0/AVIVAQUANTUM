import tensorflow as tf
import json
import math
import random
from tensorflow.keras.utils import Sequence

def listshuffle(l):
    # shuffles a list and returns a new list (not in place and does not reference the old list)
    # probably quite memory consuming
    l_new = [e for e in l]
    random.shuffle(l_new)
    return l_new

class TextSequence(Sequence):
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
        
        batch_x = tf.transpose(tf.random.shuffle(tf.transpose(batch_x)))

        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.index)
