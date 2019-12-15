import re
import whois
import csv
import datetime
from urllib.parse import urlparse

# Used to evaluate each url based on certain characteristics 

def has_at_symbol(url):
    x = re.search("@", url)
    if x:
        return 1
    return 0

def is_https(url):
    x = re.search("^https", url)
    if x:
        return 1
    return 0

def check_domain(url):
    uri = urlparse(url)
    domain = str(uri.netloc)

    if len(domain.split('-')) > 1:
        return 1
    return 0


def check_long_urls(url):
    if len(url) > 54:
        return 1
    return 0

def has_sub_domain(url_string):
    all_matches = re.findall(r'\.', url_string)
    if len(all_matches) < 3:
        return 0
    else:
        return 1

def has_ip_addreess(url):
    ip = re.search('\d\d\d\.\d\d\.\d\d\d\.\d\d\d', url)
    hex_ip = re.search('0x[\dabcdef]', url)

    if ip or hex_ip:
        indicator = 1
    else:
        indicator = 0
    return indicator

# Checks if domain exists and it's age. Younger domains are likely phishing. 
def check_whois(url_string):
    try:
        uri = urlparse(url_string)
        domain = str(uri.netloc)
        domain_info = whois.query(domain)
        creation_date = domain_info.creation_date
        d = datetime.datetime(2019, 12, 1)

        if (d - creation_date).days >= 365:
            return 1,0
        else:
            return 1,1
    except:
        return 0,1

# process and open/read the url data and write them into a CSV file
def process_file(file_name):
    attribute_extraction_funcs = [is_https, has_at_symbol, check_long_urls, check_domain, has_sub_domain, has_ip_addreess]

    print("file_name", file_name)
    f = open("dataset/" + file_name, "r")
    with open('dataset/results-' + file_name.replace(".txt", "") + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        all_results = []
        lines_processed = 0
        for line in f:
            line = line.strip().replace('"', "")
            result = [attr_func(line) for attr_func in attribute_extraction_funcs]
            f1, f2 = check_whois(line)
            result.append(f1)
            result.append(f2)
            all_results.append(result)
            print(result)
            print("Lines processed:", lines_processed)
            lines_processed += 1
        writer.writerows(all_results)
    # close the files
    csvfile.close()
    f.close()

if __name__ == "__main__":
    f = open("dataset/data_filenames.txt", "r")
    for line in f:
        process_file(line.strip())

    f.close()
