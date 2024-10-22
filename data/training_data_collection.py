import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024
        
        print(f"Starting download from {url}")
        print(f"Total file size: {total_size / (1024 * 1024):.2f} MB")
        
        with open(save_path, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024 * 1024, 
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                progress_bar.update(size)
        
        print(f"Downloaded file saved at {save_path}")
        print(f"File size on disk: {os.path.getsize(save_path) / (1024 * 1024):.2f} MB")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
    return True

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped files to {extract_to}")
    os.remove(zip_path)
    print(f"Deleted zip file at {zip_path}")

def main():
    url = 'https://figshare.com/ndownloader/files/48018562'
    download_path = '../capsule-vision-2024/data/downloaded_file.zip'
    data_path = '../capsule-vision-2024/data'
    
    # os.makedirs(os.path.dirname(download_path), exist_ok=True)
    
    if download_file(url, download_path):
        unzip_file(download_path, data_path)
    else:
        print("Download failed. Skipping unzip.")

if __name__ == '__main__':
    main()
