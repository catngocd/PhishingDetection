import csv

def preprocess(csv_file, is_phishing):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        features = list(reader)
    
    return features

if __name__ == "__main__":
    # preprocess()