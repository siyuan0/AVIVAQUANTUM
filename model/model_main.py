import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.d1 = tf.keras.layers.Dense(300, activation='relu')
        self.d2 = tf.keras.layers.Dense(100, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.d3 = tf.keras.layers.Dense(50, activation='relu')
        self.d4 = tf.keras.layers.Dense(50, activation='relu')
        self.d5 = tf.keras.layers.Dense(50, activation='relu')
        self.d6 = tf.keras.layers.Dense(50, activation='relu')
        self.dfinal = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        # x_abstract , x_label = tf.split(x, [50, 15], 1)
        # x_abstract = self.d1_abstract(x_abstract)
        # x_label = self.d1_label(x_label)
        # x = tf.concat([x_abstract,x_label],1)
        x = self.d1(inputs)
        x = self.dropout1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.dfinal(x)
        return x