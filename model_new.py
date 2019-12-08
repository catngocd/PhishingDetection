import tensorflow as tf
import numpy as np
import random
from preprocessing import preprocess_all

class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()
        self.batch_size = 1
        self.epochs = 1
        self.learning_rate = .001
        self.hidden_size = 300

        # self.Dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        # self.Dense2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        # self.Dense3 = tf.keras.layers.Dense(1, activation='softmax')
        
        # self.model.add(tf.keras.layers.Embedding(self.hidden_size))
        # self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        # self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        # self.model.add(tf.keras.layers.Dense(1))

        self.W1 = tf.Variable(tf.random.truncated_normal([8, 300], stddev=0.1)) # 2nd dimension is arbitrary
        self.b1 = tf.Variable(tf.random.truncated_normal([300], stddev=0.1)) # dimension has to match 2nd dimension of W1
        self.W2 = tf.Variable(tf.random.truncated_normal([300, 300], stddev=0.1))  # 1st dimension has to match 2nd dimension of W1
        self.b2 = tf.Variable(tf.random.truncated_normal([300], stddev=0.1)) # dimension has to match 2nd dimension of W2
        self.W3 = tf.Variable(tf.random.truncated_normal([300, 1], stddev=0.1)) # 2nd dim is 2 because we have 2 classes
        self.b3 = tf.Variable(tf.random.truncated_normal([1], stddev=0.1)) 

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        l1 = tf.matmul(tf.cast(inputs, tf.float32), self.W1) + self.b1
        l1_a = tf.nn.relu(l1)

        # print("l1:", l1_a.shape)
        l2 = tf.matmul(tf.cast(l1_a, tf.float32), self.W2) + self.b2
        l2_a = tf.nn.relu(l2)
        # print("l2:", l2_a.shape)
        l3 = tf.matmul(tf.cast(l2_a, tf.float32), self.W3) + self.b3
        l3_a = tf.nn.softmax(l3)
        # print("l3:", l3_a.shape)
        
        return l3_a

    def loss(self, logits, labels):
        # return tf.Variable(10)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        predicted_labels = tf.argmax(logits, 1)
        correct_predictions = tf.equal(predicted_labels, labels)

        return tf.reduce_sum(correct_predictions)


def train(model, train_data, train_labels):
    # num_examples = train_inputs.shape[0]
    # for i in range(num_examples // model.batch_size):
    #     batch_start_index = i * model.batch_size
    #     # take min so that we don't go past the last row of data
    #     batch_end_index = min(batch_start_index + model.batch_size, num_examples)
    #     batch_inputs = train_inputs[batch_start_index:batch_end_index,:]
    #     batch_labels = train_labels[batch_start_index:batch_end_index]
    #     with tf.GradientTape() as tape:
    #         logits = model.call(batch_inputs)
    #         loss = model.loss(logits, batch_labels)
        
    #     print("trainable: ", model.trainable_variables)
    #     grads = tape.gradient(loss, model.trainable_variables)
    #     model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    
    for start, end in zip(range(0, len(train_data) - model.batch_size, model.batch_size), 
                            range(model.batch_size, len(train_data), model.batch_size)):
        print(start, "out of", len(train_data))
        train_X = train_data[start:end]
        # print("Train X: ", train_X.shape)
        train_Y = train_labels[start:end]
        # print("Train Y: ", train_Y.shape)
        with tf.GradientTape() as tape:
            logits = model.call(train_X)
            loss = model.loss(logits, train_Y)
        # print("HERE 1")
        gradients = tape.gradient(loss, model.trainable_variables)
        # print("HERE 2")
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