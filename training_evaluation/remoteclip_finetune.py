"""
remoteclip_finetune.py
====================
OBJECTIVE: Use manually labeled images to fine-tune weights of the 
RemoteCLIP ViT-L/14 base model.

IF USING LABELED IMAGES: No prerequisite steps must be completed to
successfully run this script, besides setting directory paths.

ELSE: Run code in "RoofNetxRemoteCLIP_Classify.ipynb" to generate 
roofing image labels for further analysis.

Saves:
  best_clip_model_balanced.pth, the fine-tuned model weights.

Usage:
    python remoteclip_finetune.py

"""


# IMPORTS
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from IPython.display import display
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import open_clip


# === Load Dataset & Numerically Label ===
ROOFNET_SUBSET_DIR  = '../roofnet_data_split/train' # Insert your path to the train directory here
CSV_PATH = "../resources/roofnet_metadata_train.csv"
df = pd.read_csv(CSV_PATH)
class_names = sorted(df['material_class'].unique())
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
df['label'] = df['material_class'].apply(lambda x: class_to_idx[x])


# Take roofnet_metadata.csv and add prompts
# === MATERIAL DESCRIPTIONS ===
material_descriptions = {
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
df['prompt'] = df['material_class'].map(material_descriptions)
df = df[df['split']=='train']
df.to_csv('../resources/roofnet_metadata_train.csv')


# @title Load packages and RemoteCLIP download model weights
# Models from, Code adapted from https://github.com/ChenDelong1999/RemoteCLIP?tab=readme-ov-file
model_name = 'ViT-L-14'
checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints')
print(f'{model_name} is downloaded to {checkpoint_path}.')

model, preprocess_train, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
ckpt = torch.load(f"{checkpoint_path}/RemoteCLIP-{model_name}.pt", map_location="cpu")
message = model.load_state_dict(ckpt)
print(message)


#@title Fine-tune RemoteCLIP for 5 epochs using class rebalancing
# Check for GPU availability and define the computation device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define the local filesystem path to save the trained model weights
SAVE_PATH = os.path.join(ROOFNET_SUBSET_DIR, "../results/best_clip_model_balanced.pth")

# === Dataset Definition ===
class RoofDataset(Dataset):
    """Custom Dataset for loading roof images, their associated text prompts, and material labels."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Identify unique material categories (e.g., 'asphalt', 'metal')
        self.classes = self.data['material_class'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        # Load image and ensure 3-channel RGB format
        image = Image.open(img_path).convert("RGB")
        
        # Apply data augmentations/preprocessing (scaling, cropping, etc.)
        if self.transform:
            image = self.transform(image)
            
        # Return the processed image, the text description, and the category name
        return image, row['prompt'], row['material_class']

# === Compute class weights ===
# Calculate frequency of each class to handle dataset imbalance
class_counts = df['label'].value_counts().sort_index()
weights = 1.0 / class_counts  # Inverse frequency: rarer classes get higher weights

# Map the weight of each individual sample based on its class
sample_weights = df['label'].map(weights).values

# Initialize sampler to ensure each batch has a balanced representation of classes
sampler = WeightedRandomSampler(
    weights=sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True
)

# Initialize the dataset and the DataLoader with the balanced sampler
train_dataset = RoofDataset(CSV_PATH, ROOFNET_SUBSET_DIR, transform=preprocess_train)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

# === Training Setup ===
# Using AdamW optimizer which handles weight decay better for Transformer-based models
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# CrossEntropy with label smoothing to prevent the model from becoming overconfident
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Temperature scaling factor for the CLIP contrastive loss (standard is 0.07)
temperature = 0.07
best_loss = float("inf")

# === Training Loop ===
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, texts, labels_str in tqdm(train_loader):
        # Move data to the active device (GPU/CPU)
        images = images.to(device)
        tokenized_texts = tokenizer(texts).to(device)
        
        # Convert string labels to numerical indices for evaluation/tracking
        labels = torch.tensor([class_to_idx[l] for l in labels_str], device=device)

        # Forward Pass: Extract visual and textual embeddings
        image_features = model.encode_image(images)
        text_features = model.encode_text(tokenized_texts)
        
        # Normalize features to lie on the unit hypersphere
        normalized_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity matrix between all images and all texts in the batch
        text_features_t = normalized_text_features.T.contiguous()
        logits_per_image = (normalized_image_features @ text_features_t) / temperature
        logits_per_text = logits_per_image.T.contiguous()
        
        # Ground truth is the diagonal (image i matches text i)
        targets = torch.arange(len(images), device=device)
        
        # Symmetrical Contrastive Loss: loss from both image-to-text and text-to-image perspectives
        loss = (loss_fn(logits_per_image, targets) + loss_fn(logits_per_text, targets)) / 2

        # Backward Pass: Update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Log performance and save the model if validation (average loss) improves
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"New best model saved at epoch {epoch} with loss {avg_loss:.4f}")