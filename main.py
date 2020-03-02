import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.datafactory import TextSequence, dataloader
from model.model_main import MyModel
import json

# Create a callback that saves the model's weights every 5 epochs
checkpoint_path = 'model/checkpoints/cp-{epoch:04d}.ckpt'
model_path = 'model/model1'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=10)

model = MyModel()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

train_gen, val_data = dataloader()

model.fit(x=train_gen, 
          epochs=30,
          validation_data=val_data,
          callbacks=[cp_callback])

# model.save(model_path)