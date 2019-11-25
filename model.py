import tensorflow as tf
import numpy as np

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

def train(model, train_data, train_labels):
    pass

def test(model, test_data, test_labels):
    pass

def main():
    pass

if __name__ == "__main__":
    main()