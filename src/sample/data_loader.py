import os
import requests
import zipfile

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded file saved at {save_path}")

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped files to {extract_to}")
    os.remove(zip_path)
    print(f"Deleted zip file at {zip_path}")

def main():
    url = 'https://figshare.com/ndownloader/files/48018562'
    download_path = 'downloaded_file.zip'
    
    download_file(url, download_path)
    unzip_file(download_path, os.getcwd())

if __name__ == '__main__':
    main()
