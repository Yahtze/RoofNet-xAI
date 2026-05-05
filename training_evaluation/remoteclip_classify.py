"""
remoteclip_classify.py
====================
OBJECTIVE: Use fine-tuned weights of the RemoteCLIP ViT-L/14 base model to 
classify materials in new roofing images, sort them into directories, and 
update a metadata CSV with the predicted classes.

NOTE: We **highly** suggest running verify_material_classes.py on the 
results in order to validate the plausibility of the model predictions.

Usage:
    python remoteclip_classify.py --classification_dir <path> --results_dir <path> --model_weights <path> --metadata_csv <path>
"""

import os
import torch
import shutil
import argparse
import pandas as pd
from torchvision import transforms
from PIL import Image
from pathlib import Path
import open_clip

# List of target material categories (labels) for classification
MATERIAL_CLASSES = [
    "Thatch", "StoneSlates", "ClayTiles", "AsphaltTiles",
    "ConcreteTiles", "WoodTiles", "MetalSheetMaterials", "PolycarbonateSheetMaterials",
    "GlassSheetMaterials", "AmorphousConcrete", "AmorphousAsphalt",
    "AmorphousMembrane", "AmorphousFabric", "Unknown", "GreenVegetative"
]

def build_prompts(city_name):
    """Generates descriptive text prompts to compare against images."""
    prompts = [f"{material} in {city_name}" for material in MATERIAL_CLASSES]
    return prompts

def extract_city_name_from_filename(filename):
    """Parses the filename to identify the city name."""
    base = Path(filename).stem
    if '-' in base:
        city_part = base.split('-')[0]
        return city_part.replace('_', ' ').title()
    elif 'height' in base:
        city_part = base.split('_height')[0]
        return city_part.replace('_', ' ').title()
    elif 'imsat' in base:
        city_part = base.split('_imsat')[0]
        return city_part.replace('_', ' ').title()
    return base

def predict_and_move(image_path, city_name, model, tokenizer, preprocess, device, output_base_dir):
    """Performs inference on an image, moves the file to the predicted material folder, and returns the prediction."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    area = width * height

    if area <= 1000:
        print(f"{os.path.basename(image_path)} too small ({area} px), skipping.")
        return None

    image = preprocess(image).unsqueeze(0).to(device)
    prompts = build_prompts(city_name)
    tokenized_prompts = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokenized_prompts)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (100.0 * image_features @ text_features.T).squeeze(0)
    best_idx = similarities.argmax().item()
    predicted_material = MATERIAL_CLASSES[best_idx]

    target_dir = os.path.join(output_base_dir, predicted_material)
    try:
        shutil.copy2(image_path, target_dir)
        print(f"{os.path.basename(image_path)} classified as {predicted_material} and moved.")
    except Exception:
        print(f"{os.path.basename(image_path)} already present in {predicted_material}.")
        
    return predicted_material

def main(classification_dir, results_dir, model_weights_path, metadata_csv):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the CSV metadata
    print(f"Loading metadata from {metadata_csv}...")
    try:
        df = pd.read_csv(metadata_csv)
    except UnicodeDecodeError:
        # Fallback to latin-1 if there are special characters in city names
        df = pd.read_csv(metadata_csv, encoding='latin-1')
        
    if 'material_class' not in df.columns:
        df['material_class'] = None
    df['material_class'] = df['material_class'].astype(object)

    # Initialize model
    print("Loading RemoteCLIP model...")
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Create subdirectories for every material class
    for material in MATERIAL_CLASSES:
        os.makedirs(os.path.join(results_dir, material), exist_ok=True)

    print(f"Processing images in {classification_dir}...")
    for img_file in os.listdir(classification_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(classification_dir, img_file)
            city_name = extract_city_name_from_filename(img_file)
            
            # Predict and move
            predicted_material = predict_and_move(full_path, city_name, model, tokenizer, preprocess, device, results_dir)
            
            # Update DataFrame
            if predicted_material:
                mask = df['filename'] == img_file
                if mask.any():
                    df.loc[mask, 'material_class'] = predicted_material
                else:
                    print(f"  ⚠️ Warning: {img_file} not found in {os.path.basename(metadata_csv)}. Skipping CSV update for this file.")

    # Overwrite the CSV with the new material_class column included
    print(f"Saving updated metadata back to {metadata_csv}...")
    df.to_csv(metadata_csv, index=False)
    print("Process complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify roof materials using RemoteCLIP.")
    parser.add_argument('--classification_dir', type=str, required=True, help="Directory containing cropped images to classify")
    parser.add_argument('--results_dir', type=str, default='classified_roofs', help="Output directory to store sorted folders")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to .pth file containing classification model weights")
    parser.add_argument('--metadata_csv', type=str, required=True, help="Path to the CSV containing roof metadata")
    
    args = parser.parse_args()
    main(args.classification_dir, args.results_dir, args.model_weights, args.metadata_csv)