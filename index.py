from extract_and_recomend import ImageFeatureExtractor
from backend.resize import extract_filename
from utils import extract_info_from_url
from PIL import Image
import os

# Usage of the class
def display_images(image_paths):
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img.show()  # This opens the image in the default image viewer
        except Exception as e:
            print(f"Failed to display {img_path}: {e}")

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

def main():

    url = 'https://static.zara.net/photos///2024/V/0/2/p/0679/402/250/2/w/2048/0679402250_3_1_1.jpg?ts=1704357326556'
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


if __name__ == '__main__':
    main()