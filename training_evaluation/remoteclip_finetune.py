"""
remoteclip_finetune.py
====================
OBJECTIVE: Use manually labeled images to fine-tune weights of the 
RemoteCLIP ViT-L/14 base model.

Usage:
    python remoteclip_finetune.py --train_dir <path> --csv_path <path> --save_path <path> [--epochs 5] [--batch_size 32]
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import open_clip

# === MATERIAL DESCRIPTIONS ===
MATERIAL_DESCRIPTIONS = {
    "Thatch": "thatched roof (dried grasses / straw or palm)",
    "GreenVegetative": "roof with vegetation on it",
    "StoneSlates": "dark stone slate roof",
    "ClayTiles": "tiled clay / tiled ceramic roof",
    "AsphaltTiles": "angled asphalt shingle roof",
    "ConcreteTiles": "tiled concrete / tiled cement roof",
    "WoodTiles": "wood shingle roof",
    "MetalSheetMaterials": "corrugated or tiled metal roof (silver / dark / painted)",
    "PolycarbonateSheetMaterials": "polycarbonate roof",
    "GlassSheetMaterials": "glass roof (clear or mirrored)",
    "AmorphousConcrete": "flat concrete roof",
    "AmorphousAsphalt": "asphalt-coated roof (bitumen layer or rolled roofing)",
    "AmorphousMembrane": "membrane roof (bright EPDM/TPO)",
    "AmorphousFabric": "tensile fabric roof (PVC / PTFE / canvas)",
    "Unknown": "unknown material, image may be too low resolution or obstructed"
}

class RoofDataset(Dataset):
    """Custom Dataset for loading roof images, their associated text prompts, and material labels."""
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform
        self.classes = self.data['material_class'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['full_path'] # Now uses the absolute, recursively-found path
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row['prompt'], row['material_class']

def main(train_dir, csv_path, save_path, epochs, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare DataFrame and Labels
    print("Preparing dataset metadata...")
    df = pd.read_csv(csv_path)
    
    # Strip hidden whitespace from strings
    df['filename'] = df['filename'].astype(str).str.strip()

    class_names = sorted(df['material_class'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    df['label'] = df['material_class'].apply(lambda x: class_to_idx[x])
    
    df['prompt'] = df['material_class'].map(MATERIAL_DESCRIPTIONS)
    
    # Check the state of the DataFrame before filtering
    print(f"Total rows in CSV: {len(df)}")
    
    # --- UPDATED SPLIT FILTERING ---
    if 'split' in df.columns:
        # Fill any completely blank cells with 'unknown'
        df['split'] = df['split'].fillna('unknown')
        
        # Create a mask to keep both 'train' and 'unknown'
        valid_splits = ['train', 'unknown']
        mask = df['split'].astype(str).str.lower().isin(valid_splits)
        
        train_count = mask.sum()
        print(f"Found {train_count} rows marked for training ('train' or 'unknown').")
        
        # Apply the filter
        df = df[mask]
    else:
        print("No 'split' column found. Proceeding with all rows.")
        
    if len(df) == 0:
        raise ValueError("The DataFrame is empty AFTER filtering splits! Double check the values in your 'split' column.")

    # Extract just the file name (and stem) from the CSV to avoid folder path mismatches
    df['clean_name'] = df['filename'].apply(lambda x: Path(x).name)
    df['clean_stem'] = df['filename'].apply(lambda x: Path(x).stem)

    # --- RECURSIVE IMAGE LOOKUP ---
    print(f"Scanning '{train_dir}' and subdirectories for images...")
    image_lookup = {}
    for p in Path(train_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_lookup[p.name] = str(p)
            image_lookup[p.stem] = str(p)
            
    print(f"Found {len(image_lookup) // 2} valid images in the directory.")

    # Try matching the exact filename first (image.jpg). If that fails, try matching the stem (image).
    df['full_path'] = df['clean_name'].map(image_lookup).fillna(df['clean_stem'].map(image_lookup))
    
    missing_count = df['full_path'].isna().sum()
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} images listed in the CSV were not found in '{train_dir}'.")
        print("Here are the first 5 missing files the script was looking for:")
        print(df[df['full_path'].isna()]['clean_name'].head(5).tolist())
        print("Dropping missing images from this training session...")
        df = df.dropna(subset=['full_path'])
        
    if len(df) == 0:
        raise ValueError("No matching images found! The filenames in the CSV do not match the files on disk.")
    
    # Load Model
    print("Loading RemoteCLIP base model...")
    model_name = 'ViT-L-14'
    checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints')
    model, preprocess_train, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # Load using the direct path and weights_only=True to prevent warnings/errors
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device)

    # Class weights for sampler
    class_counts = df['label'].value_counts().sort_index()
    weights = 1.0 / class_counts 
    sample_weights = df['label'].map(weights).values

    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    train_dataset = RoofDataset(df, transform=preprocess_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    temperature = 0.07
    best_loss = float("inf")

    print(f"Starting training for {epochs} epochs...")
    if os.path.dirname(save_path): 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, texts, labels_str in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            tokenized_texts = tokenizer(texts).to(device)
            labels = torch.tensor([class_to_idx[l] for l in labels_str], device=device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(tokenized_texts)
            
            normalized_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_features_t = normalized_text_features.T.contiguous()
            logits_per_image = (normalized_image_features @ text_features_t) / temperature
            logits_per_text = logits_per_image.T.contiguous()
            
            targets = torch.arange(len(images), device=device)
            
            loss = (loss_fn(logits_per_image, targets) + loss_fn(logits_per_text, targets)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with loss {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RemoteCLIP for roof material classification.")
    parser.add_argument('--train_dir', type=str, required=True, help="Directory containing the training images (searched recursively)")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to roofnet_metadata_train.csv")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the best weights (e.g., best_clip_model_balanced.pth)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training (default: 32)")
    
    args = parser.parse_args()
    main(args.train_dir, args.csv_path, args.save_path, args.epochs, args.batch_size)