import tensorflow as tf
import numpy as np
import random
from preprocessing import preprocess_all

class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()
        self.batch_size = 1
        self.epochs = 1
        self.learning_rate = .005
        self.hidden_size = 30

        # with batch size = 1, epochs = 1, learning rate = .005, hidden size = 30
        # 

        # self.Dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        # self.Dense2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        # self.Dense3 = tf.keras.layers.Dense(1, activation='softmax')
        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Embedding(self.hidden_size))
        self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # self.E = tf.keras.layers.Embedding(self.hidden_size )
        # self.L1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        # self.L2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        # self.L3 = tf.keras.layers.Dense(2, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):

        # embeddings = self.E(inputs)
        # L1_output = self.L1(inputs)
        # L2_output = self.L2(L1_output)
        # L3_output = self.L3(L2_output)

        return self.model(inputs)

    def loss(self, logits, labels):
        # return tf.Variable(10)
        #labels = tf.one_hot(labels,1)
        return tf.reduce_mean(self.loss_f(tf.cast(labels, tf.float32), tf.cast(logits, tf.float32)))

    def accuracy(self, logits, labels):
        logits = tf.squeeze(logits)
        x = np.sum(tf.equal(tf.math.round(logits), labels))
        return x
        # predicted_labels = tf.argmax(logits, 1)
        # correct_predictions = tf.equal(predicted_labels, labels)

        # return tf.reduce_sum(correct_predictions)


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
        print(start+1, "out of", len(train_data))
        train_X = train_data[start:end]
        # print("Train X: ", train_X.shape)
        # train_Y = tf.expand_dims(train_labels[start:end], 1)
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
    
    return num_correct / len(test_labels)

def main():
    csv_files = ["results-phishing_url.csv", "results-cc_1_first_9617_urls.csv"]
    is_phishing = [True, False]
    # print(preprocess_all(csv_files, is_phishing))
    train_data, train_labels, test_data, test_labels = preprocess_all(csv_files, is_phishing)
    print("train_data", train_data.shape)
    print("train_labels", train_labels.shape)
    print("test_data", test_data.shape)
    print("test_labels", test_labels.shape)

    # train_data = np.random.choice([0,1], size=(400,8))
    # train_labels = np.random.choice([0,1], size=(400,1))
    # test_data = np.random.choice([0,1], size=(200,8))
    # test_labels = np.random.choice([0,1], size=(200,1))

    model = Model()

    for epoch in range(model.epochs):
        train(model, train_data, train_labels)

    test_accuracy = test(model, test_data, test_labels)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()