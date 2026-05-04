# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of BRAILS.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.

# Major modifications made by RoofNet development team to below script.
# Above copyright attribute kept for traceability of code.

import os
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from glob import glob

# === CONFIGURATION ===
INPUT_DIR = "roofnet_gsat_imagery" # <-- Insert path to your imagery to be cropped
OUTPUT_DIR = "roofnet_gsat_imagery_cropped" # <-- Insert path to where cropped imagery should be saved

TEXT_PROMPT = "single building in middle of image without occlusion."
BOX_THRESHOLD = 0.25
DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"

# === DATA CLASSES ===

@dataclass
class FilterRoofBoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class FilterRoofDetectionResult:
    score: float
    label: str
    box: FilterRoofBoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'FilterRoofDetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=FilterRoofBoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

# === CORE DETECTION LOGIC ===

def detect(image: Image.Image, object_detector, labels: List[str], threshold: float = 0.3) -> List[FilterRoofDetectionResult]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    Accepts the pre-loaded object_detector pipeline.
    """
    # Ensure labels end with a period for DINO
    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [FilterRoofDetectionResult.from_dict(result) for result in results]

    return results

def process_and_crop_image(image_path: str, output_dir: str, object_detector):
    """
    Detects the building, extracts the largest bounding box, applies padding, 
    and saves the cropped image using the pre-loaded model.
    """
    img_name = os.path.basename(image_path)
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {img_name}: {e}")
        return False

    W, H = img.size

    # Pass the pre-loaded detector into the detect function
    detections = detect(img, object_detector, [TEXT_PROMPT], threshold=BOX_THRESHOLD)

    if len(detections) == 0:
        print(f"Skipping {img_name}: No building detected above threshold.")
        return False

    # Extract boxes
    boxes = [det.box.xyxy for det in detections]

    if len(boxes) > 1:
        box_areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
        box_idx = np.argmax(box_areas)
    else:
        box_idx = 0

    box = boxes[box_idx]
    
    x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    # Apply padding
    x0, y0 = max(1, x0-40), max(1, y0-40)
    x1, y1 = min(W-1, x1+40), min(H-1, y1+40)
    
    crop = img.crop((x0, y0, x1, y1))
    crop.save(os.path.join(output_dir, img_name), 'PNG')
    
    return True

# === MAIN EXECUTION ===

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_files = glob(os.path.join(INPUT_DIR, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images in '{INPUT_DIR}'.")
    print("Loading Grounding DINO model into memory (this happens only once)...")
    
    # 1. Determine the best available hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load the model weights into RAM/VRAM just ONE time
    global_detector = pipeline(
        model=DETECTOR_ID, 
        task="zero-shot-object-detection", 
        device=device
    )
    print(f"Model loaded successfully on device: {device}\n")
    
    success_count = 0
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}...")
        
        # 3. Pass the loaded model to the processing function
        if process_and_crop_image(img_path, OUTPUT_DIR, global_detector):
            success_count += 1
            
    print(f"\n✅ Done! Successfully cropped {success_count} out of {len(image_files)} images.")