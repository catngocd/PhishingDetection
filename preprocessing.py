import csv
import numpy as np

def preprocess_all(csv_files, is_phishing):
    '''
    Inputs:
        csv_file: a list of strings where each string is the name of a csv file 
        is_phishing: a Boolean array where each boolean represents whether the file contains phishing link
    Output: (shuffled features from all csv files, shuffled labels of all csv files)
    '''
    all_features = []
    all_labels = []
    for i, csv_file in enumerate(csv_files):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            features = list(reader)
            all_features += features
            if is_phishing[i]:
                all_labels += [1]*len(features)
            else:
                all_labels += [0]*len(features)
        f.close()
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    all_features = shuffle_all(all_features)
    all_labels = shuffle_all(all_labels)

    return all_features, all_labels

def shuffle_all(features, labels):
    '''
    Inputs:
        features: a list of list of feautres
        labels: a list of all the labels
    Output: (shuffled features, shuffled labels)
    '''
    indices = np.array(range(len(labels)))
    np.random.shuffle(indices)

    features = features[indices]
    labels = labels[indices]

    return features, labels

if __name__ == "__main__":
    # preprocess()
