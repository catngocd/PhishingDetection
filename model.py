import tensorflow as tf
import numpy as np
import random

class Model(tf.keras.model):
    def __init__(self):

        super(Model, self).__init__()
        self.batch_size = None
        self.epochs = None
        self.learning_rate = .001
        self.hidden_size = 300

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.hidden_size))
        self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        return self.model(inputs)

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        pass


def train(model, train_data, train_labels, num_features):
    # train data will be a list of lists
    M = []
    state_size = len(train_data)
    action_size = 2
    Q = np.random.random_sample((state_size,action_size))
    # num of episodes
    K = len(train_data)
    target_val_function = np.copy(Q)
    counter = 0



def test(model, test_data, test_labels):
    pass

def main():
    pass

if __name__ == "__main__":
    main()
