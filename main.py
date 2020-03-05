import tensorflow as tf
import os
import datetime
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.datafactory import TextSequence, dataloader
from model.model_main import MyModel, AccMetrics
import json

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Create a callback that saves the model's weights every 5 epochs
checkpoint_path = 'model/checkpoints/cp-{epoch:04d}.ckpt'
model_path = 'model/model1'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=100)
log_dir = os.path.join('log')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(0.01,
                                                              decay_steps=10000,
                                                              decay_rate=0.96,
                                                              staircase=True)


dl = dataloader(10)
for train_gen, train_data, val_data in iter(dl):
    print("training on new fold")
    model = MyModel(dl.word_index)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_scheduler),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    # Acc_func = AccMetrics(train_data, val_data, model)
    model.fit(x=train_gen, 
            epochs=1000,
            validation_data=val_data,
            callbacks=[cp_callback, tensorboard_callback],
            use_multiprocessing = False)
    
    x_val, y_val = val_data
    y_val_pred = model(x_val)
    
    model.save('model\save_model')
    break

# model.save(model_path)