import os
import torch
from torchvision import models, transforms
from PIL import Image

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to inference mode

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0)

def crop_and_save(image_path, output_dir, model):
    image_tensor = load_image(image_path)
    predictions = model(image_tensor)
    
    # Assuming the first detected object is the clothing item
    # This might need tuning based on the outputs you see
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    highest_score_index = scores.argmax()
    box = boxes[highest_score_index]

    # Convert tensor to list
    box = box.tolist()

    # Crop the image
    image = Image.open(image_path)
    cropped_image = image.crop((box[0], box[1], box[2], box[3]))
    cropped_image = cropped_image.resize((299, 299), Image.LANCZOS)

    # Prepare output path and save
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, base_name)
    cropped_image.save(output_path)
    print(f"Saved cropped image to {output_path}")


def process_directory(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):  # Check for JPEG images
            file_path = os.path.join(input_dir, filename)
            try:
                crop_and_save(file_path, output_dir, model)
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")

# Example usage
input_directory = 'img'
output_directory = 'cropped'
process_directory(input_directory, output_directory)