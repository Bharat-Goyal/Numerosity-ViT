import os
from PIL import Image
from transformers import ViTImageProcessor
import torch

def fetch_images(directory):
    image_dict = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    # Iterate over immediate subdirectories
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            image_list = []
            # Iterate over images in the subdirectory
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                
                # Check if it's an image file
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(file_path).convert("RGB")
                    image_list.append(image)
            # Add the image list to the dictionary with the subdirectory name as the key
            image_dict[subdir] = processor(images=image_list, return_tensors="pt").to(device).pixel_values
    #Returns experimentwise images, which have already been downsamples to ViT input range. 
    return image_dict

if __name__ == '__main__': 
    directory = '/home/hice1/bgoyal7/scratch/HML/experiment_data/exp1_equal_area_circles'
    image_dict = fetch_images(directory)
    # print(image_dict)
