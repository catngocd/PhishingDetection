import pandas as pd

data = pd.read_csv("csv/verified_online.csv")

def quotes(url):
    return url.replace('"', "")
    # if url[0] == "\"":
    #     return url[1:-1]
    # else:
    #     return type(url)

data["url"] = data["url"].apply(quotes)


data.to_csv(path_or_buf="csv/phishing_url.txt", columns=["url"], header=False, index=False)