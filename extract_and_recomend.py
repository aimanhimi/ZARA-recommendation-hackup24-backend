import os
import numpy as np
import json
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as kimage
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ImageFeatureExtractor:
    def __init__(self):
        base_model = ResNet50(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        print("Model loaded and ready for feature extraction.")

    def extract_features(self, img_path):
        img = kimage.load_img(img_path)
        img_array = kimage.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array)
        return features.flatten()

    def load_features(self, jsonfile):
        with open(jsonfile, 'r') as f:
            data = json.load(f)
        feature_list = np.array(data['features']).squeeze()
        image_paths = data['image_paths']
        return feature_list, image_paths

    def find_similar_images(self, input_features, feature_list, image_paths, top_k=6):
        similarities = cosine_similarity([input_features], feature_list)
        top_indices = np.argsort(similarities[0])[::-1][1:top_k+1]

        image_ids = [image_paths[i].split('\\')[-1].split('.')[0].replace('_', '/') for i in top_indices]

        # Load the CSV file into a DataFrame and flatten it into a single column
        df = pd.read_csv('data.csv', header=None, usecols=[0])
        df = df.values.flatten()
        df = pd.DataFrame(df, columns=['url'])

        # Filter the DataFrame to only include rows where the URL contains one of the IDs
        # Filter the DataFrame to only include rows where the URL contains one of the IDs
        matching_rows = df[df['url'].apply(lambda url: any(id in url for id in image_ids) if pd.notnull(url) else False)]
        # Print the matching URLs
        url_values = matching_rows['url'].values.tolist()

        # Remove duplicates by converting the list to a set, then convert it back to a list
        unique_url_values = list(set(url_values))
        
        # return [image_paths[i] for i in top_indices]
        return unique_url_values