import os
import pandas as pd
from PIL import Image
from io import BytesIO
import requests

def extract_filename(url):
    # Extracts a part of the URL to use as the filename
    # Example URL: https://static.zara.net/photos///2024/V/0/3/p/5767/521/712/2/w/2048/5767521712_6_1_1.jpg?ts=1707751045954
    # Extract '5767_521_712' from the URL
    parts = url.split('/')
    # Typically, the parts we want are on index 11,12,13 in the parts vector, so:
    return parts[11] + '_' + parts[12] + '_' + parts[13]
def download_and_process_images(df, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for index, row in df.iterrows():
        url = row['IMAGE_VERSION_3']  # Ensure this column name matches your dataset

        try:
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = crop_center(img)
                img = img.resize((224, 224), Image.ANTIALIAS)
                filename = extract_filename(url) + '.jpg'
                img.save(os.path.join(folder_path, filename))
        except:
                pass

def crop_center(img):
    width, height = img.size
    new_width = new_height = min(width, height)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img.crop((left, top, right, bottom))


