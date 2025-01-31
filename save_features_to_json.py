import os
import numpy as np
import json
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# Load the pre-trained ResNet50 modelfeatures.json
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

def extract_features(img_path, model):
    img = kimage.load_img(img_path, target_size=(224, 224))
    img_array = kimage.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

image_dir = 'resize/wom_winter'
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
feature_list = [extract_features(img_path, model) for img_path in image_paths]

# Convert list of arrays to a single NumPy array
feature_array = np.array(feature_list)
#Convert array of shape (479, 1, 2048) to (479, 2048)
feature_array = feature_array.reshape(feature_array.shape[0], feature_array.shape[2])

# Save features and image paths
features_path = 'featuresKW.json'
with open(features_path, 'w') as f:
    # Convert the entire array to list before saving
    json.dump({'features': feature_array.tolist(), 'image_paths': image_paths}, f)