from http.client import HTTPException
import os
from typing import Optional, Union

import pandas as pd
from fastapi import FastAPI # type: ignore
from fastapi.middleware.cors import CORSMiddleware
from similarism import get_image_caption_from_id, search_similar_captions
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from extract_and_recomend import ImageFeatureExtractor
from resize import extract_filename
from utils import extract_info_from_url


app = FastAPI()


origins = [
    "http://localhost:3000",  # React development server
    "http://localhost:8000",  # FastAPI server (if serving static files)
]

# Add CORS middleware to allow specific origins (or use "*" for all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Adjust the list according to your needs
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# df = pd.read_csv('data.csv', skiprows=1, header=None)

df = pd.read_csv('image_ids_and_urls.csv')


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.get("/items/")
def get_items(page: Optional[int] = 1):
    # Set the number of items per page
    items_per_page = 8

    # Calculate the start and end indices for the items on the requested page
    start = (page - 1) * items_per_page
    end = start + items_per_page

    # If the start index is out of bounds, default to page 1
    if start < 0 or start >= len(df):
        start = 0
        end = items_per_page

    # Get only the desired items
    products_data = df.iloc[start:end]

    # Prepare the dictionary to hold the product data
    products = {}
    for index, row in products_data.iterrows():
        # Create a product key for each row
        product_key = f'product{index+1}'
        # Assuming the URLs are in columns 0, 1, and 2 and these are all strings or numbers
        products[product_key] = [str(row[1])]
    
    return products


@app.post("/get_similar_items")
def get_similar_items(selected_product_id: int):
    
    search_term = get_image_caption_from_id(selected_product_id)


    results =  search_similar_captions(search_term)


    for result in results:
        # Assuming 'result' is your tuple and 'df' is your DataFrame
        id, _, _ = result  # Unpack the tuple to get the ID
        row = df.loc[df['id'] == int(id)]  # Get the row from the DataFrame
        # Print the row
        print(row['url'].values[0])


    return results

def select_json(season, demographic):

    if season == 'W' and demographic == '3':
        jsonfile = 'featuresKW.json'
        folder_path = 'resize/kds_winter'
    elif season == 'W' and demographic == '2':
        jsonfile = 'featuresMW.json'
        folder_path = 'resize/men_winter'
    elif season == 'W' and demographic == '1':
        jsonfile = 'featuresWW.json'
        folder_path = 'resize/wom_winter'
    elif season == 'V' and demographic == '3':
        jsonfile = 'featuresKV.json'
        folder_path = 'resize/kds_summer'
    elif season == 'V' and demographic == '2':
        jsonfile = 'featuresMV.json'
        folder_path = 'resize/men_summer'
    elif season == 'V' and demographic == '1':
        jsonfile = 'featuresWV.json'
        folder_path = 'resize/wom_summer'
    else:
        raise Exception("Invalid season or demographic.")

    return jsonfile, folder_path

@app.post("/get_similar_items_from_url")
def get_similar_items_from_url(selected_product_id: int):
    # Check if the index is within the valid range
    if selected_product_id <= 0 or selected_product_id > len(df):
        raise HTTPException(status_code=404, detail="Product index out of range")

    # Fetch the specific product data using the provided index
    try:
        row = df.iloc[selected_product_id-1]
        url = str(row[1])  # Convert all URLs to strings
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # print(url[0])
    # return
    filename = extract_filename(url)+'.jpg'
    print(filename)

    season, demography = extract_info_from_url(url)
    jsonfile, folder_path = select_json(season, demography)

    extractor = ImageFeatureExtractor()
    feature_list, image_paths = extractor.load_features(jsonfile)
    input_features = extractor.extract_features(folder_path+'/'+filename)
    recommended_images = extractor.find_similar_images(input_features,
                                                       feature_list,
                                                       image_paths)
    
    # display_images(recommended_images)

    print("Recommended Images:", recommended_images)

    res = []

    for img_path in recommended_images:
        res.append(img_path)

    return {"recommended_images": res }

@app.get("/product/{product_index}")
def get_item(product_index: int):
    # Check if the index is within the valid range
    if product_index <= 0 or product_index > len(df):
        raise HTTPException(status_code=404, detail="Product index out of range")

    # Fetch the specific product data using the provided index
    try:
        row = df.iloc[product_index-1]
        product_urls = [str(row[1])]  # Convert all URLs to strings
        return {"product": product_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
