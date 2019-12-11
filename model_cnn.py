import tensorflow as tf
import numpy as np
from preprocess_cnn import convert_urls_to_vector


class Model(tf.keras.Model):
    def __init__(self, vocab_size, kernel_s):

        """
        The Model class predicts whether a url is phishing or benign.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique characters in the data
        """

        super(Model, self).__init__()

        self.learning_rate = 0.0001
        self.vocab_size = vocab_size
        self.embedding_size = 32
        self.batch_size = 64
        self.hidden_layer_size_1 = 512
        self.hidden_layer_size_2 = 256
        self.hidden_layer_size_3 = 128

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=200))
        self.model.add(tf.keras.layers.Reshape((200, 32, 1)))
        self.model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(kernel_s, self.embedding_size), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(200 - kernel_s + 1, 1), strides=(1,1), padding='valid'))
        self.model.add(tf.keras.layers.Dense(self.hidden_layer_size_1, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.hidden_layer_size_2, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.hidden_layer_size_3, activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        return self.model(inputs)

    def loss(self, logits, labels):
        return self.loss_func(labels, logits)


    def accuracy(self, logits, labels):
        predicted_labels = tf.squeeze(tf.where(logits>0.5, 1, 0))
        correct_predictions = tf.equal(predicted_labels, labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    counter = 0
    batch_low = 0
    batch_high = model.batch_size
    while (batch_high <= len(train_inputs)):
        print(batch_low, "out of", len(train_inputs))
        batch_inputs = np.array(train_inputs[batch_low : batch_high])
        batch_labels = np.array(train_labels[batch_low : batch_high])
        with tf.GradientTape() as tape:
            predictions = model.call(batch_inputs)
            batch_labels = tf.expand_dims(batch_labels, 1)
            loss = model.loss(predictions, batch_labels)
        print("Loss:", loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        counter += 1
        batch_low = (model.batch_size * counter)
        batch_high = (model.batch_size * counter) + model.batch_size

def test(model, test_inputs, test_labels):
    counter = 0
    batch_low = 0
    batch_high = model.batch_size
    accuracy = 0.0
    while (batch_high <= len(test_inputs)):
        inputs = np.array(test_inputs[batch_low : batch_high])
        labels = np.array(test_labels[batch_low : batch_high])
        test_answers = model.call(inputs)
        accuracy += model.accuracy(test_answers,labels)
        counter += 1
        batch_low = (model.batch_size * counter)
        batch_high = (model.batch_size * counter) + (model.batch_size)
    return accuracy/counter

def main():
    file_names = ["dataset/phishing_url.txt", "dataset/cc_1_first_9617_urls"]
    is_phishing = [True, False]
    train_data, train_labels, test_data, test_labels, vocabulary = convert_urls_to_vector(file_names, is_phishing)
    model = Model(len(vocabulary), 5)
    # TODO: Set-up the training step
    for i in range(0, 10):
        train(model, train_data, train_labels)
    # TODO: Set up the testing steps
    accuracy = test(model, test_data, test_labels)
    # Print out accuracy
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
    main()
