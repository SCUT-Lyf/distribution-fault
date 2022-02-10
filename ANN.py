# from make_datasets import make_datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import numpy as np


def plot_acc(history, epochs, val_freq):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['sparse_categorical_accuracy']
    val_accuracy = history.history['val_sparse_categorical_accuracy']
    # val_loss = history.history['val_mean_squared_logarithmic_error']
    # val_mean_squared_logarithmic_error = history.history['val_mean_squared_logarithmic_error']
    fig, ax = plt.subplots(2, 1, sharex="col")
    plt.subplots_adjust(hspace=0.5)
    ax[0].plot(loss, label='loss')
    ax[0].plot(np.arange(0, epochs, val_freq), val_loss, label='val_loss')
    ax[0].set_ylim(np.min(loss)*0.6, np.max(loss)*1.2)
    ax[0].set_title('training and validation loss')
    ax[0].legend()

    ax[1].plot(accuracy, label='accuracy')
    ax[1].plot(np.arange(0, epochs, val_freq), val_accuracy, label='val_accuracy')
    ax[1].set_ylim(np.min(accuracy)*0.9, 1.05)
    ax[1].set_title('training and validation accuracy')
    ax[1].legend()
    ax[1].set_xlabel("epochs")
    plt.show()

class ANN_model(Model):
    def __init__(self, n_class):
        super(ANN_model, self).__init__()
        self.n_class = n_class
        self.LS1 = tf.keras.layers.GRU(32,  return_sequences=True, input_shape=(400, 6))
        self.LS2 = tf.keras.layers.GRU(64,  return_sequences=True)
        # self.d1 = tf.keras.layers.Dropout(0.2)
        self.L0 = tf.keras.layers.GRU(128,  return_sequences=True)
        self.L1 = tf.keras.layers.GRU(128,  return_sequences=True)
        self.L2 = tf.keras.layers.GRU(32, return_sequences=True)
        self.L3 = tf.keras.layers.GRU(6,  return_sequences=False)
        self.last = tf.keras.layers.Dense(self.n_class)
        # self.f2 = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.LS1(inputs)
        x = self.LS2(x)
        # x = self.d1(x)
        x = self.L0(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        y = self.last(x)
        return y




if __name__ == '__main__':
    npzfile = np.load('../datasets/datasets1_standard.npz')
    datasets = npzfile["datasets"]
    labels = npzfile["labels"].reshape(-1, 1).astype(np.int64)
    batchs = 64
    epoch = 200
    n_class = 4
    counts = np.bincount(labels[:, 0])
    class_weight = {0:1/counts[0], 1:1/counts[1], 2:1/counts[2], 3:1/counts[3]}
    x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, stratify=labels, random_state=1000)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batchs).prefetch(tf.data.experimental.AUTOTUNE)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(batchs).prefetch(tf.data.experimental.AUTOTUNE)
    model = ANN_model(n_class)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
                  , metrics=['sparse_categorical_accuracy']
                  # , metrics=['accuracy']
                  , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                  # , loss=tf.keras.losses.MeanAbsoluteError()
                  )
    history = model.fit(train_db, epochs=epoch, validation_data=test_db, validation_freq=10
                        # , class_weight=class_weight
                        )
    # model.build(input_shape=(None, 400, 6))
    model.summary()
    plot_acc(history, epochs=epoch, val_freq=10)
    tf.saved_model.save(model, '/TrainedModel')
    # print(history.history['val_sparse_categorical_accuracy'][-1])
    print("最大准确率", np.max(history.history['val_sparse_categorical_accuracy']))
    print("平均准确率:", np.mean(history.history['val_sparse_categorical_accuracy'][-5:]))






