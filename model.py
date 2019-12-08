import tensorflow as tf
import numpy as np
import random
from preprocessing import preprocess_all

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 10
        self.num_classes = 2
        self.input_size = 8
        self.epochs = 200
        self.learning_rate = .001
        self.hidden_size = 300
        self.W = np.zeros((self.input_size,self.num_classes))
        self.b = np.zeros(self.num_classes)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        input_with_w = np.matmul(inputs,self.W) + self.b
        e = np.exp(input_with_w)
        predictions = e / np.sum(e, axis=1, keepdims=True)
        return predictions

    def loss(self, logits, labels):
        return  np.sum(- np.log(logits[range(0,len(logits)), labels]))/logits.shape[0]

    def accuracy(self, logits, labels):
        return np.mean(np.argmax(logits, axis=1) == labels)

    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. The learning
        algorithm for updating weights and biases mentioned in
        class works for one image, but because we are looking at
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        delta = probabilities.copy()
        delta[range(len(delta)), labels] -= 1
        delta_L_b = np.sum(delta, axis=0) / self.batch_size
        delta_L_w = np.matmul(inputs.T, delta) / self.batch_size
        return delta_L_w, delta_L_b

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W = self.W - self.learning_rate * gradW
        self.b = self.b - self.learning_rate * gradB


def train(model, train_data, train_labels):
    counter = 0
    batch_low = 0
    batch_high = model.batch_size
    while (batch_high <= len(train_data)):
        inputs = train_data[batch_low : batch_high]
        labels = train_labels[batch_low : batch_high]
        probabilities = model.call(inputs)
        print("Loss:", model.loss(probabilities, labels))
        delta_L_w, delta_L_b = model.back_propagation(inputs, probabilities, labels)
        model.gradient_descent(delta_L_w, delta_L_b)
        counter += 1
        batch_low = (model.batch_size * counter)
        batch_high = (model.batch_size * counter) + model.batch_size
    if (batch_low < len(train_data)):
        inputs = train_data[batch_low : len(train_data)]
        labels = train_labels[batch_low : len(train_data)]
        probabilities = model.call(inputs)
        model.back_propagation(inputs, probabilities, labels)


def test(model, test_data, test_labels):
    test_answers = model.call(test_data)
    return model.accuracy(test_answers,test_labels)

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
