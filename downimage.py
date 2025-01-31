from PIL import Image
import requests
from io import BytesIO
import os
from urllib.parse import urlparse


with open("inditextech_hackupc_challenge_images.csv", "r") as f:

    lines = f.readlines()

    counter = 0

    for line in lines:

        images = line.split(",")

        for i in range(len(images)):
            images[i] = (images[i].strip('"'))

        
        response = requests.get(images[0])
        # Open the image using PIL
        image = Image.open(BytesIO(response.content))

        image.save(r"C:\Users\joelc\OneDrive - UAB\Joel\HACKUPC24\img\"{str(counter)}.jpg")

        counter += 1