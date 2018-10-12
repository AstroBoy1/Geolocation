import pandas as pd
import urllib.request
import mysql.connector
from mysql.connector import errorcode


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
    urllib.request.urlretrieve(url, 'metadataPhoto.jpg')


def main():
    read_sql()
    # download_image()


if __name__ == "__main__":
    main()
