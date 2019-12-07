import tensorflow as tf
import numpy as np
import random
from preprocessing import preprocess_all

class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()
        self.batch_size = 128
        self.epochs = 1
        self.learning_rate = .001
        self.hidden_size = 300

        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Embedding(self.hidden_size))
        self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='softmax'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        return self.model(inputs)

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        predicted_labels = tf.argmax(logits, 1)
        correct_predictions = tf.equal(predicted_labels, labels)

        return tf.reduce_sum(correct_predictions)


def train(model, train_data, train_labels):
    for start, end in zip(range(0, len(train_data) - model.batch_size, model.batch_size), 
                            range(model.batch_size, len(train_data), model.batch_size)):
        train_X = train_data[start:end]
        train_Y = train_labels[start:end]
        with tf.GradientTape() as tape:
            logits = model.call(train_X)
            loss = model.loss(logits, np.array(train_Y))
        print("HERE 1")
        gradients = tape.gradient(loss, model.trainable_variables)
        print("HERE 2")
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_data, test_labels):
    num_correct = 0
    for start, end in zip(range(0, len(test_data) - model.batch_size, model.batch_size), 
                        range(model.batch_size, len(test_data), model.batch_size)):
        test_X = test_data[start:end]
        test_Y = test_labels[start:end]

        logits = model.call(test_X)
        num_correct += model.accuracy(logits, test_Y)
    
    return num_correct / test_data.shape[0]

def main():
    csv_files = ["results-phishing_url.csv", "results-cc_1_first_9617_urls.csv"]
    is_phishing = [True, False]
    train_data, train_labels, test_data, test_labels = preprocess_all(csv_files, is_phishing)

    model = Model()

    for epoch in range(model.epochs):
        train(model, train_data, train_labels)

    test_accuracy = test(model, train_data, train_labels)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
