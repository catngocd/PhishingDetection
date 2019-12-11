import model_vanilla_nn as vanillaModule
import model_cnn as cnnModule
from preprocessing_vanilla_nn import preprocess_all
from preprocess_cnn import convert_urls_to_vector
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Taking in respective files
    csv_files_vanilla = ["dataset/results-phishing_url.csv", "dataset/results-cc_1_first_9617_urls.csv"]
    file_names_cnn = ["dataset/phishing_url.txt", "dataset/cc_1_first_9617_urls"]
    is_phishing = [True, False]
    # Training and Testing inputs and labels for Vanilla NN
    train_data_vanilla, train_labels_vanilla, test_data_vanilla, test_labels_vanilla = preprocess_all(csv_files_vanilla, is_phishing)
    # Training and Testing inputs and labels for CNN Module
    train_data_cnn, train_labels_cnn, test_data_cnn, test_labels_cnn, vocabulary = convert_urls_to_vector(file_names_cnn, is_phishing)
    # Initialization
    model_1 = vanillaModule.Model()
    # Model 2 is initialized later according to the graph requirements
    model_1_accuracies = []
    model_2_accuracies = []

    #NOTE: UNCOMMENT THE FOLLOWING IF YOU WANT ACCURY BY KERNEL SIZE FOR THE CNN MODEL.

    # kernel_sizes = range(3,7)
    # for kernel_size in kernel_sizes:
    #     model_2 = cnnModule.Model(len(vocabulary),kernel_size)
    #     for i in range(0, 10):
    #         cnnModule.train(model_2, train_data_cnn, train_labels_cnn)
    #     accuracy_cnn = cnnModule.test(model_2, test_data_cnn, test_labels_cnn)
    #     model_2_accuracies.append(accuracy_cnn.numpy())
    # plt.scatter(kernel_sizes, model_2_accuracies, edgecolors='r')
    # plt.xlabel('Kernel Sizes')
    # plt.ylabel('CNN Model Accuracy')
    # plt.title('Accuracy by Kernel Sizes')
    # plt.show()


    # NOTE: UNCOMMENT THE FOLLOWING IF YOU WANT ACCURACY BY EPOCH FOR BOTH MODELS!

    model_2 = cnnModule.Model(len(vocabulary),5)
    for epoch in range(model_1.epochs):
        vanillaModule.train(model_1, train_data_vanilla, train_labels_vanilla)
        cnnModule.train(model_2, train_data_cnn, train_labels_cnn)
        accuracy_vanilla = vanillaModule.test(model_1, test_data_vanilla, test_labels_vanilla)
        model_1_accuracies.append(accuracy_vanilla)
        accuracy_cnn = cnnModule.test(model_2, test_data_cnn, test_labels_cnn)
        model_2_accuracies.append(accuracy_cnn)
    # Print out accuracy
    print("Accuracy using Vanilla NN:", accuracy_vanilla)
    print("Accuracy using CNN: ", accuracy_cnn)
    # data to plot
    n_groups = 10
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, model_1_accuracies, bar_width,
    alpha=opacity,
    color='r',
    label='Vanilla RNN')

    rects2 = plt.bar(index + bar_width, model_2_accuracies, bar_width,
    alpha=opacity,
    color='g',
    label='CNN')

    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by epochs')
    plt.xticks(index + bar_width, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
