import tensorflow as tf
import numpy as np
import random
from preprocessing_vanilla_nn import preprocess_all

class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()
        self.batch_size = 1
        self.epochs = 1
        self.learning_rate = .005
        self.hidden_size = 30
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        return self.model(inputs)

    def loss(self, logits, labels):
        return tf.reduce_mean(self.loss_f(tf.cast(labels, tf.float32), tf.cast(logits, tf.float32)))

    def accuracy(self, logits, labels):
        logits = tf.squeeze(logits)
        x = np.sum(tf.equal(tf.math.round(logits), labels))
        return x


def train(model, train_data, train_labels):
    for start, end in zip(range(0, len(train_data) - model.batch_size, model.batch_size),
                            range(model.batch_size, len(train_data), model.batch_size)):
        print(start+1, "out of", len(train_data))
        train_X = train_data[start:end]
        train_Y = train_labels[start:end]
        with tf.GradientTape() as tape:
            logits = model.call(train_X)
            loss = model.loss(logits, train_Y)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_data, test_labels):
    num_correct = 0
    for start, end in zip(range(0, len(test_data) - model.batch_size, model.batch_size),
                        range(model.batch_size, len(test_data), model.batch_size)):
        test_X = test_data[start:end]
        test_Y = test_labels[start:end]

        logits = model.call(test_X)
        num_correct += model.accuracy(logits, test_Y)
    return num_correct / len(test_labels)

def main():
    csv_files = ["dataset/results-phishing_url.csv", "dataset/results-cc_1_first_9617_urls.csv"]
    is_phishing = [True, False]
    train_data, train_labels, test_data, test_labels = preprocess_all(csv_files, is_phishing)
    model = Model()
    for epoch in range(model.epochs):
        train(model, train_data, train_labels)
    test_accuracy = test(model, test_data, test_labels)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
