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
        self.d1_abstract = tf.keras.layers.Dense(500, activation='relu')
        self.d1_label = tf.keras.layers.Dense(200, activation='relu')
        self.d1 = tf.keras.layers.Dense(500, activation='relu') #test to delete
        self.d2 = tf.keras.layers.Dense(300, activation='relu')
        self.d3 = tf.keras.layers.Dense(300, activation='relu')
        self.d4 = tf.keras.layers.Dense(150, activation='relu')
        self.d5 = tf.keras.layers.Dense(150, activation='relu')
        self.d6 = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        # x_abstract , x_label = tf.split(x, [50, 15], 1)
        # x_abstract = self.d1_abstract(x_abstract)
        # x_label = self.d1_label(x_label)
        # x = tf.concat([x_abstract,x_label],1)
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        return x

model = MyModel()
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

abstract_tensor = tf.convert_to_tensor(abstract_sequence, dtype=tf.float32)
label_tensor = tf.convert_to_tensor(label_sequence, dtype=tf.float32)
x_train = tf.concat([abstract_tensor,label_tensor],1)
signatures_category = [0 if s<50 else 1 for s in signatures]
y_train = tf.convert_to_tensor(signatures_category, dtype=tf.float32)
y_train = tf.reshape(y_train, (y_train.shape[0],1))


model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2, shuffle=True)
