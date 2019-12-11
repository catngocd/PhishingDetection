import collections
import numpy as np

def shuffle_all(features, labels):
    '''
    Inputs:
        features: a list of list of feautres
        labels: a list of all the labels
    Output: (shuffled features, shuffled labels)
    '''
    indices = np.array(range(len(labels)))
    np.random.shuffle(indices)

    features = features[indices,:]
    labels = labels[indices]

    return features, labels

def convert_urls_to_vector(file_names, is_phishing):

    def count_chars(file_name):
        all_text = ""
        for file_name in file_names: 
            file = open(file_name, 'r')
            all_text += file.read()
            file.close()
        
        char_counts = collections.Counter(all_text)
        char_counts.pop('\n')

        return char_counts

    char_counts = count_chars(file_names)

    vector_length = 200

    url_vectors = []
    labels = []
    char_to_id = {}
    
    # add <PAD> and <UNK> to char_to_id
    char_to_id["<PAD>"] = 0
    char_to_id["<UNK>"] = 1
    id = 2
    for file_id, file_name in enumerate(file_names):
        num_urls_in_file  = 0
        f = open(file_name, 'r')
        for line in f:
            num_urls_in_file += 1
            line = line.strip().replace('"', "") # remove " " that surrounds some URLs
            url = line.strip('\n')
            url_vec = np.full(vector_length , 0) # Fill with <PAD>
            for i in range(min(vector_length, len(url))):
                char = url[i]
                if char_counts[char] >= 100:
                    if char not in char_to_id:
                        char_to_id[char] = id
                        id += 1
                    url_vec[i] = char_to_id[char]
                else:
                    url_vec[i] = 1 # <UNK>
            url_vectors.append(url_vec)
        
        # Generate labels
        if is_phishing[file_id]:
            labels += [1]*num_urls_in_file 
        else:
            labels += [0]*num_urls_in_file 


    # Shuffle
    url_vectors, labels = shuffle_all(np.array(url_vectors), np.array(labels))

    # Train-test split
    train_ratio = 0.8
    num_urls = len(url_vectors)
    split_index = int(train_ratio * num_urls)

    train_data = np.array(url_vectors[0:split_index])
    train_labels = np.array(labels[0:split_index])

    test_data = np.array(url_vectors[split_index:])
    test_labels = np.array(labels[split_index:])

    return train_data, train_labels, test_data, test_labels, char_to_id

            

def main():
    # urls_file_name = 
    file_names = ["dataset/phishing_url.txt", "dataset/cc_1_first_9617_urls"]
    is_phishing = [True, False]

    convert_urls_to_vector(file_names, is_phishing)

    # save vector representations
    # write_to = urls_file_name.replace(".txt", "") + ".csv"
    # np.savetxt(write_to, url_vectors, fmt='%s', delimiter=",")
    # print("Finished writing to", write_to)





if __name__ == "__main__":
    main()