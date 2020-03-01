import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
# data prep
with open('training_data.json', 'rb') as f:
    training_doc = json.load(f)
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts([t['abstract']['_value'] + t['label']['_value'] for t in training_doc])
word_index = tokenizer.word_index # creates the index to convert words to numbers

abstract_text=[]
label_text=[]
signatures=[]
abstract_maxlen = 50
label_maxlen = 15

for t in training_doc:
    abstract_text.append(t['abstract']['_value'])
    label_text.append(t['label']['_value'])
    signatures.append(t['numberOfSignatures'])
abstract_sequence = tokenizer.texts_to_sequences(abstract_text)
abstract_sequence = pad_sequences(abstract_sequence, maxlen=abstract_maxlen, padding='post', truncating='post')
label_sequence = tokenizer.texts_to_sequences(label_text)
label_sequence = pad_sequences(label_sequence, maxlen=label_maxlen, padding='post', truncating='post')

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1_abstract = tf.keras.layers.Dense(50, input_shape=(50,), activation='relu')
        self.d1_label = tf.keras.layers.Dense(15, input_shape=(15,), activation='relu')
        self.d2 = tf.keras.layers.Dense(30, activation='sigmoid')
        self.d3 = tf.keras.layers.Dense(1, activation='relu')
    def call(self, x):
        x = tf.concat([self.d1_abstract(x[0:50]),self.d1_label(x[50:])],1)
        x = self.d2(x)
        x = self.d3(x)
        return x

model = MyModel()
loss_fn = tf.keras.losses.MeanAbsoluteError()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mse'])

abstract_tensor = tf.convert_to_tensor(abstract_sequence, dtype=tf.float32)
label_tensor = tf.convert_to_tensor(label_sequence, dtype=tf.float32)
x_train = tf.concat([abstract_tensor,label_tensor],1)
y_train = tf.convert_to_tensor(signatures, dtype=tf.float32)
print(tf.shape(x_train))
print(tf.shape(y_train))

model.fit(x_train, y_train, batch_size=32, epochs=5)
