import requests
import os

def get_xml_files(url, counter, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for i in counter:
        try:
            response = requests.get(url[i])
            with open(result_dir + "/" + str(i) + '.xml', 'w+') as f1:
                f1.write(str(response.content))
        except:
            continue

def process_file(file_name):
    urls = []
    f = open("dataset/" + file_name, "r")
    for line in f:
        line = line.strip()
        urls.append(line)
    counters = range(0, len(urls))
    get_xml_files(urls, counters, os.getcwd() + "/dataset/xml_extractions/" + file_name)


if __name__ == "__main__":
    f = open("dataset/" + "data_filenames.txt", "r")
    for line in f:
        process_file(line.strip())
