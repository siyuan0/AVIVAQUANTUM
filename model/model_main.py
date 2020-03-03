import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # base dense layers
        self.base = []
        for i in range(3):
            self.base += [tf.keras.layers.Dense(300, activation='relu')]
        
        # feature layers
        self.feature = []
        for i in range(3):
            self.feature += [tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
                             tf.keras.layers.MaxPool1D(3)]

        self.dfinal = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = inputs  

        for k, l in enumerate(self.base):
            x = l(x)
        
        x = tf.expand_dims(x,2) # prep for conv1d

        for k, l in enumerate(self.feature):
            x = l(x)

        x = self.dfinal(x)

        return x

    