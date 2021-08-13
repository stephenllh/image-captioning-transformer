from urllib.request import urlopen
import zipfile
from io import BytesIO


def download_and_unzip(url, extract_to="../input"):
    http_response = urlopen(url)
    z = zipfile.ZipFile(BytesIO(http_response.read()))
    z.extractall(path=extract_to)


if __name__ == "__main__":
    flickr_dataset_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    flickr_text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    download_and_unzip(flickr_dataset_url)
    download_and_unzip(flickr_text_url)
