import collections
import numpy as np

def count_chars(file_name):
    file = open(file_name, 'r')
    text = file.read()
    char_counts = collections.Counter(text)
    char_counts.pop('\n')
    return char_counts

def convert_urls_to_vector(file_name, char_counts):
    vector_length = 200
    url_vectors = []
    char_to_id = {}
    id = 1
    f = open(file_name, 'r')
    for line in f:
        url = line.strip('\n')
        url_vec = np.full(vector_length , "<PAD>")
        for i in range(min(vector_length, len(url))):
            char = url[i]
            if char_counts[char] >= 100:
                if char not in char_to_id:
                    char_to_id[char] = id
                    id += 1
                url_vec[i] = char_to_id[char]
            else:
                url_vec[i] = "<UNK>"
        url_vectors.append(url_vec)
    return url_vectors
            

def main():
    # urls_file_name = "dataset/phishing_url.txt"
    urls_file_name = "dataset/cc_1_first_9617_urls"
    # get character counts
    char_counts = count_chars(urls_file_name)
    # convert URLs to vector representation
    url_vectors = convert_urls_to_vector(urls_file_name, char_counts)

    # save vector representations
    write_to = urls_file_name.replace(".txt", "") + ".csv"
    np.savetxt(write_to, url_vectors, fmt='%s', delimiter=",")
    print("Finished writing to", write_to)





if __name__ == "__main__":
    main()