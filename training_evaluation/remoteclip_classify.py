"""
remoteclip_classify.py
====================
OBJECTIVE: Use fine-tuned weights of the RemoteCLIP ViT-L/14 base model to 
classify materials in new roofing images.

NOTE: We **highly** suggest running verify_material_classes.py on the 
results in order to validate the plausibility of the model predictions.

Saves:
  Series of classified images to predicted material folder.

Usage:
    python remoteclip_classify.py

"""

# === IMPORTS === #
import os
import torch
import shutil
from torchvision import transforms
from PIL import Image
from pathlib import Path
import open_clip

# === CONFIGURATION ===
# Path to the source directory containing images that need to be categorized
CLASSIFICATION_DIR = '' 

# Path where the results and sorted material folders will be stored
RESULTS_PATH = os.path.join(CLASSIFICATION_DIR, '../results')

# Path to the fine-tuned model weights (.pth file)
model_weights_path = 'path/to/your/model.pth' 

# The base directory where classified images will be moved
output_base_dir = RESULTS_PATH 

# Set calculation device: uses GPU (cuda) if available, otherwise defaults to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# List of target material categories (labels) for classification
material_classes = [
    "Thatch", "StoneSlates", "ClayTiles", "AsphaltTiles",
    "ConcreteTiles", "WoodTiles", "MetalSheetMaterials", "PolycarbonateSheetMaterials",
    "GlassSheetMaterials", "AmorphousConcrete", "AmorphousAsphalt",
    "AmorphousMembrane", "AmorphousFabric", "Unknown", "GreenVegetative"
]

# Initialize the CLIP model (Vision Transformer L/14) with pre-trained weights
model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')

# Image preprocessing pipeline to format images for the CLIP model:
# 1. Resize to 224x224 pixels
# 2. Convert to a numerical tensor
# 3. Normalize using standard CLIP mean and standard deviation values
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

# Load the specific tokenizer for the ViT-L-14 model architecture
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# Load the custom fine-tuned weights into the model
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)

# Set model to evaluation mode to ensure consistent predictions
model.eval()

# Create subdirectories for every material class if they do not already exist
for material in material_classes:
    os.makedirs(os.path.join(output_base_dir, material), exist_ok=True)

# === FUNCTIONS ===

def build_prompts(city_name):
    """Generates descriptive text prompts to compare against images."""
    prompts = [f"{material} in {city_name}" for material in material_classes]
    return prompts

def extract_city_name_from_filename(filename):
    """
    Parses the filename to identify the city name. 
    Handles variations like 'City-ID.png' or 'City_heightXX.png'.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    base = Path(filename).stem
    if '-' in base:
        city_part = base.split('-')[0]
        city_name = city_part.replace('_', ' ').title()
        return city_name
    elif 'height' in base:
        city_part = base.split('_height')[0]
        city_name = city_part.replace('_', ' ').title()
        return city_name
    elif 'imsat' in base:
        city_part = base.split('_imsat')[0]
        city_name = city_part.replace('_', ' ').title()
    return city_name

def already_classified(img_name):
    """Checks if the image has already been moved to one of the result folders."""
    for material in material_classes:
        target_path = os.path.join(output_base_dir, material, img_name)
        if os.path.exists(target_path):
            return True
    return False

def predict_and_move(image_path, city_name):
    """
    Performs inference on an image and moves the file to the predicted material folder.
    """
    # Load image and convert to standard RGB
    image = Image.open(image_path).convert("RGB")

    # Reject images that are too small to provide meaningful features
    width, height = image.size
    area = width * height

    if area <= 1000:
        print(f"{os.path.basename(image_path)} too small ({area} px), skipping.")
        return

    # Prepare the image for the model
    image = preprocess(image).unsqueeze(0).to(device)

    # Convert text prompts into numerical tokens
    prompts = build_prompts(city_name)
    tokenized_prompts = tokenizer(prompts).to(device)

    # Calculate feature vectors for both image and text
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokenized_prompts)

    # Normalize vectors so the comparison (dot product) reflects similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores and identify the best match
    similarities = (100.0 * image_features @ text_features.T).squeeze(0)
    best_idx = similarities.argmax().item()
    predicted_material = material_classes[best_idx]

    # Relocate the file to the directory named after the predicted material
    target_dir = os.path.join(output_base_dir, predicted_material)
    try:
      shutil.move(image_path, target_dir)
      print(f"{os.path.basename(image_path)} classified as {predicted_material} and moved.")
    except:
      print(f"{os.path.basename(image_path)} already present.")

# === MAIN LOOP ===
# Loop through the files in the classification directory and process each image
for material in material_classes:
  input_images_dir = os.path.join(CLASSIFICATION_DIR)

  for img_file in os.listdir(input_images_dir):
      # Process only valid image file types
      if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
          full_path = os.path.join(input_images_dir, img_file)
          # Extract location metadata for more accurate prompt matching
          city_name = extract_city_name_from_filename(img_file)
          # Execute prediction and file organization
          predict_and_move(full_path, city_name)