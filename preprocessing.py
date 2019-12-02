import csv
import numpy as np

def preprocess_all(csv_files, is_phishing):
    '''
    csv_file: list of strings that is the csv_file name
    is_phishing: boolean array
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

    return all_features, all_labels

if __name__ == "__main__":
    # preprocess()