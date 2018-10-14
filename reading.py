import pandas as pd
import urllib.request
import mysql.connector
from mysql.connector import errorcode
import csv
import unittest


def read_sql():
    try:
        cnx = mysql.connector.connect(user='root', host='195.206.104.109', passwd='1fbDKLcEvKfBiItK')
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        cnx.close()


def download_image():
    df = pd.read_csv('photo_metadata.csv', nrows=10)
    farm_id = str(df['flickr_farm'][0])
    server_id = str(df['flickr_server'][0])
    identity = str(df['id'][0])
    secret = str(df['flickr_secret'][0])
    url = "https://farm" + farm_id + ".staticflickr.com/" + server_id + "/" + identity + "_" + secret + ".jpg"
    # urllib.request.urlretrieve(url, 'metadataPhoto.jpg')


def bulk_url_builder(chunk=100000):
    """Lags at chunk size 1M, 100k is good"""
    reader = pd.read_csv('photo_metadata.csv', chunksize=chunk)
    count = 0
    with open('urls.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "url"])
        for df in reader:
            for i in range(len(df)):
                farm_id = str(df['flickr_farm'][i + count])
                server_id = str(df['flickr_server'][i + count])
                identity = str(df['id'][i + count])
                secret = str(df['flickr_secret'][i + count])
                url = "https://farm" + farm_id + ".staticflickr.com/" + server_id + "/" + identity + "_" + secret + ".jpg"
                # urllib.request.urlretrieve(url, str(count + i) + ".jpg")
                writer.writerow([identity, url])
            count += chunk
            print(count)


def url_check(file):
    df = pd.read_csv(file, nrows=10)
    return df['id']


# class MyTest(unittest.TestCase):
#     def test(self):
#         self.assertEqual(url_check("urls.csv"), url_check("photo_metadata.csv"))


def main():
    # read_sql()
    # bulk_url_builder()
    return 1


if __name__ == "__main__":
    main()
