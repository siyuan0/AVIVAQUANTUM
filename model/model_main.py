import tensorflow as tf
from tensorflow.python.ops import math_ops
import tensorflow.python.ops.metrics_impl
from tensorflow.python.ops.metrics_impl import mean
from tensorflow.python.eager import context
from tensorflow.keras.callbacks import Callback

class MyModel(tf.keras.Model):
    def __init__(self, word_index=None):
        super(MyModel, self).__init__()
        
        # base dense layers
        self.base = []
        self.base = [tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)),
                     tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)),
                     tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.003)),]
        
        # feature layers
        self.feature = []
        for i in range(3):
            self.feature += [tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer='l2'),
                             tf.keras.layers.Dropout(0.3)]
        # self.feature += [tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        #                  tf.keras.layers.Dropout(0.3)]
        if word_index is not None: self.embedding = tf.keras.layers.Embedding(len(word_index.keys())+1,32)
        self.maxpool = tf.keras.layers.MaxPool1D(3)
        self.flatten = tf.keras.layers.Flatten()
        self.lnorm = tf.keras.layers.LayerNormalization()
        self.dinitial = tf.keras.layers.Dense(100, activation='softmax')
        self.dfinal1 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = inputs
        
        # x = tf.expand_dims(x,-1) # prep for conv1d
        x = self.embedding(x)

        for k, l in enumerate(self.feature):
            x = l(x)
        x = self.flatten(x)
        x = self.lnorm(x)
        for k, l in enumerate(self.base):
            x = l(x)
        
        x = self.dfinal1(x)
        return x
        # x2 = self.dfinal2(x)
        # return tf.concat([x1,x2],-1)

    
class AccMetrics(Callback):
    def __init__(self, train_data, val_data, model):
        self.x_train, self.y_train = train_data
        self.x_val, self.y_val = val_data
        self.model = model

    def on_epoch_end(self, *args):
        y_train_preds = self.model(self.x_train)
        y_train_preds_cat = tf.Variable([1 if e>50 else 0 for e in y_train_preds])
        y_train_true_cat = tf.Variable([1 if e>50 else 0 for e in self.y_train])
       
        print("\nTraining acc: ",tf.reduce_sum(tf.cast(tf.math.equal(y_train_preds_cat,y_train_true_cat), dtype=tf.float16))/(y_train_preds.shape[0]))

        
        y_val_preds = self.model(self.x_val)
        y_val_preds_cat = tf.Variable([1 if e>50 else 0 for e in y_val_preds])
        y_val_true_cat = tf.Variable([1 if e>50 else 0 for e in self.y_val])
        print("\nVal acc: ",tf.reduce_sum(tf.cast(tf.math.equal(y_val_preds_cat,y_val_true_cat), dtype=tf.float16))/(y_val_preds.shape[0]))
        # y_train_preds = self.model(self.x_train)
        # y_train_preds_cat = tf.math.round(y_train_preds[:,0])
        # y_train_true_cat = self.y_train[:,0]
        
        # y_val_preds = self.model(self.x_val)
        # y_val_preds_cat = tf.math.round(y_val_preds[:,0])
        # y_val_true_cat = self.y_val[:,0]
        
        # print("\nTraining acc: ",tf.reduce_sum(tf.multiply(y_train_preds_cat,y_train_true_cat))/(y_train_preds.shape[0]))
        # print("Validation acc: ",tf.reduce_sum(tf.multiply(y_val_preds_cat,y_val_true_cat))/(y_val_preds.shape[0]))


        