from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import numpy as np

class modelbuild():
    def __init__(self, model_dir, n_class):
        base_model = tf.saved_model.load(model_dir)
        self.pre_training_model = base_model(input_shape = (7,1),
                                        include_top = False,
                                        weights = 'imagenet')
        for layer in self.pre_training_model:
            layer.trainable = False
        self.L1 = tf.keras.layers.Dense(64, activation='relu')
        self.L2 = tf.keras.layers.Dense(8, activation='relu')
        self.last = tf.keras.layers.Dense(n_class)

    def predict(self,x):
        x = self.pre_training_model(x)
        x = self.L1(x)
        x = self.L2(x)
        y = self.last(x)
        return y





if __name__ == '__main__':
    model_dir = '\TrainedModel'
    new_model = modelbuild(model_dir, 11)

