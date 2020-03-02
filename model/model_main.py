import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.d1 = tf.keras.layers.Dense(300, activation='relu') #test to delete
        self.d2 = tf.keras.layers.Dense(300, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.168)
        self.d3 = tf.keras.layers.Dense(300, activation='relu')
        self.d4 = tf.keras.layers.Dense(300, activation='relu')
        self.d5 = tf.keras.layers.Dense(300, activation='relu')
        self.d6 = tf.keras.layers.Dense(150, activation='relu')
        self.d7 = tf.keras.layers.Dense(150, activation='relu')
        self.d8 = tf.keras.layers.Dense(150, activation='relu')
        self.d9 = tf.keras.layers.Dense(150, activation='relu')
        self.d10 = tf.keras.layers.Dense(150, activation='relu')
        self.d11 = tf.keras.layers.Dense(150, activation='relu')
        self.d12 = tf.keras.layers.Dense(50, activation='relu')
        self.d13 = tf.keras.layers.Dense(50, activation='relu')
        self.d14 = tf.keras.layers.Dense(50, activation='relu')
        self.d15 = tf.keras.layers.Dense(50, activation='relu')
        self.d16 = tf.keras.layers.Dense(50, activation='relu')
        self.d17 = tf.keras.layers.Dense(50, activation='relu')
        self.dfinal = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        # x_abstract , x_label = tf.split(x, [50, 15], 1)
        # x_abstract = self.d1_abstract(x_abstract)
        # x_label = self.d1_label(x_label)
        # x = tf.concat([x_abstract,x_label],1)
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.dropout1(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.dfinal(x)
        return x