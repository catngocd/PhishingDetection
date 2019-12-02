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

        self.E = tf.keras.layers.Embedding(self.hidden_size )
        self.L1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.L2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.L3 = tf.keras.layers.Dense(2, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):

        embeddings = self.E(inputs)
        L1_output = self.L1(embeddings)
        L2_output = self.L2(L1_output)
        L3_output = self.L3(L2_output)

        return L3_output

    def loss(self, logits, labels):
        pass

    def accuracy(self, logits, labels):
        pass

def rewards(rewards, labels):
    rewards_list = np.where(rewards == labels, 1, -1)
    return rewards_list

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

    # while episodes 1 up to K
    for e in range(0, K):
        state1 = train_data[0]
        # phi function stuff here
        while counter < num_features:



    return 0


def test(model, test_data, test_labels):
    pass

def main():
    pass

if __name__ == "__main__":
    main()
