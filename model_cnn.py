import tensorflow as tf
import numpy as np
from char_preprocess import convert_urls_to_vector


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size
        self.learning_rate = 0.01
        self.vocab_size = vocab_size
        self.embedding_size = 32
        self.batch_size = 64
        self.rnn_size = 128
        self.input_channel =  3
		self.hidden_layer_size_1 = 512
        self.hidden_layer_size_2 = 256
        self.hidden_layer_size_3 = 128
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN
        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.W_1 = tf.Variable(tf.random.truncated_normal(shape=[3,4,5,6], stddev=0.1))
		self.b_1 = tf.Variable(tf.random.truncated_normal(shape=[6], stddev=0.1))
        self.rnn_layer = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
        self.dense_w_1 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_layer_size_1,self.hidden_layer_size_1],stddev=0.1),dtype=tf.float32)
		self.dense_w_2 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_layer_size_1,self.hidden_layer_size_2],stddev=0.1),dtype=tf.float32)
		self.dense_w_3 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_layer_size_2,self.hidden_layer_size_3],stddev=0.1),dtype=tf.float32)
		self.dense_b_1 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_layer_size_1],stddev=0.1),dtype=tf.float32)
		self.dense_b_2 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_layer_size_2],stddev=0.1),dtype=tf.float32)
		self.dense_b_3 = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_layer_size_3],stddev=0.1),dtype=tf.float32)


    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn

        -Note 1: If you use an LSTM, the final_state will be the last two outputs of calling the rnn.
        If you use a GRU, it will just be the second output.

        -Note 2: You only need to use the initial state during generation. During training and testing it can be None.
        """
        output,final_state = self.rnn_layer(self.E(inputs))
        conv = tf.nn.conv2d(outputs, self.W_1, [3,4,5,6], "SAME")
		conv_with_bias = tf.nn.bias_add(conv,self.b_1)
		pooled_conv = tf.nn.max_pool(conv_with_bias,[2,2],[1,2,2,1],"SAME")
		dense_layer_1 = tf.matmul(pooled_conv, self.dense_w_1) + self.dense_b_1
		dense_layer_1_relu = tf.nn.relu(dense_layer_1)
		dense_layer_2 = tf.matmul(dense_layer_1_relu, self.dense_w_2) + self.dense_b_2
		dense_layer_2_relu = tf.nn.relu(dense_layer_2)
		dense_layer_3 = tf.matmul(dense_layer_2_relu, self.dense_w_3) + self.dense_b_3
        dense_layer_3_relu = tf.nn.relu(dense_layer_3)
        softmaxed = tf.nn.softmax(dense_layer_3_relu)
		return softmaxed

    def loss(self, logits, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    indices = list(range(len(train_labels)))
	shuffled_indices = tf.random.shuffle(indices)
	shuffled_inputs = tf.gather(train_inputs,shuffled_indices)
	shuffled_labels = tf.gather(train_labels, shuffled_indices)
	counter = 0
	batch_low = 0
	batch_high = model.batch_size
	while (batch_high <= len(shuffled_inputs_random)):
		batch_inputs = np.array(shuffled_inputs[batch_low : batch_high])
		batch_labels = np.array(shuffled_labels[batch_low : batch_high])
		with tf.GradientTape() as tape:
			predictions = model.call(batch_inputs)
			loss = model.loss(predictions, batch_labels)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		counter += 1
		batch_low = (model.batch_size * counter)
		batch_high = (model.batch_size * counter) + model.batch_size

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)

    """
    counter = 0
    batch_low = 0
    batch_high = model.batch_size * model.window_size
    loss = 0.0
    while (batch_high <= len(test_inputs)):
        inputs = tf.reshape(np.array(test_inputs[batch_low : batch_high]), shape=(model.batch_size, model.window_size))
        labels = tf.reshape(np.array(test_labels[batch_low : batch_high]), shape=(model.batch_size, model.window_size))
        test_answers,_ = model.call(inputs, None)
        loss += model.loss(test_answers,labels)
        counter += 1
        batch_low = (model.batch_size * model.window_size * counter)
        batch_high = (model.batch_size * model.window_size * counter) + (model.batch_size * model.window_size)
    return tf.exp(loss/counter)

def main():
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.

    # TO-DO:  Separate your train and test data into inputs and labels

    preprocessed = get_data("data/train.txt", "data/test.txt")
    train_ids = preprocessed[0]
    test_ids = preprocessed[1]
    vocabulary = preprocessed[2]
    train_inputs = np.array(train_ids[:-1])
    train_labels = np.array(train_ids[1:])
    # Test Data
    test_inputs = np.array(test_ids[:-1])
    test_labels = np.array(test_ids[1:])
    # TODO: initialize model and tensorflow variables
    model = Model(len(vocabulary))
    # TODO: Set-up the training step
    train(model, train_inputs, train_labels)
    # TODO: Set up the testing steps
    perplexity = test(model, test_inputs, test_labels)
    # Print out perplexity
    print("Perplexity: ", perplexity)

if __name__ == '__main__':
    main()
