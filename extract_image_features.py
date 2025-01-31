from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as kimage
import numpy as np


class ImageFeatureExtractor:
    def __init__(self):
        base_model = ResNet50(weights='imagenet')
        self.model = Model(inputs=base_model.input,
                           outputs=base_model.get_layer('avg_pool').output)
        print("Model loaded and ready for feature extraction.")

    def extract_features(self, img_path):
        img = kimage.load_img(img_path, target_size=(224, 224))
        img_array = kimage.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array)
        return features.flatten()

