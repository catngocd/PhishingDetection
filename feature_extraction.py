import re
import whois
import csv

from urllib.parse import urlparse

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
    domain = f"{uri.netloc}"
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

def is_unusual_url(url):
    try:
        domain = whois.query(url)
        return 0
    except:
        return 1

def has_ip_addreess(url):
    ip = re.search('\d\d\d\.\d\d\.\d\d\d\.\d\d\d', url)
    hex_ip = re.search('0x[\dabcdef]', url)

    if ip or hex_ip:
        indicator = 1
    else:
        indicator = 0
    return indicator

def is_young(url_string):
    try:
        domain = whois.query(url_string)
        creation_date = domain.creation_date
        d = datetime.date(2019, 3, 11)
        if (d - creation_date).days >= 365:
            return 0
        else:
            return 1
    except:
        return 0

attribute_extraction_funcs = [is_https, has_at_symbol, check_long_urls, check_domain, has_sub_domain, has_ip_addreess]
test_funcs = [is_young]

def process_file(file_name):
    global attribute_extraction_funcs
    global test_funcs
    f = open("dataset/" + file_name, "r")
    with open('results-' + file_name.replace(".txt", "") + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        for line in f:
            line = line.strip()
            result = [attr_func(line) for attr_func in test_funcs]
            result.insert(0, line)
            writer.writerow(result)
    
    csvfile.close() 
    f.close()

if __name__ == "__main__":
    f = open("dataset/data_filenames.txt", "r")
    for line in f:
        process_file(line.strip())
    
    f.close()