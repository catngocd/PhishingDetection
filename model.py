import tensorflow as tf
import numpy as np

class Model(tf.keras.model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None

    def call(self, inputs):
        pass

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