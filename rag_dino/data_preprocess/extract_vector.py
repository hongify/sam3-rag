import torch
from PIL import Image
import json
import os
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)

def generate_embeddings(root_path):
    with open(os.path.join(root_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    results = []

    for entry in tqdm(metadata):
        img_path = os.path.join(root_path, entry['image_path'].lstrip('./'))
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            mid_layer = 12
            patch_tokens = outputs.hidden_states[mid_layer][:, 1:, :]  
            embedding = patch_tokens.mean(dim=1).cpu().numpy().flatten()

        entry['embedding'] = embedding.tolist() 
        results.append(entry)

    with open(os.path.join(root_path, 'metadata_with_embeddings.json'), 'w') as f:
        json.dump(results, f)

    print("extract done ")

generate_embeddings('/Dataset/khanhha_edit')