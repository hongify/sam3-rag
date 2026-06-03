import torch
from PIL import Image
import json
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "openai/clip-vit-large-patch14" 
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)

def generate_embeddings(root_path):
    metadata_path = os.path.join(root_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    results = []

    print(f"🚀 Extracting CLIP embeddings (Original Image Focus)...")

    for entry in tqdm(metadata):
        # 1. 이미지 경로 설정
        img_path = os.path.join(root_path, entry['image_path'].lstrip('./'))
        
        # 2. 이미지 로드 (원본 유지)
        image = Image.open(img_path).convert("RGB")

        # --- [변경 포인트] ---
        # 기존에 있던 'if entry['is_crack']...: img_np[~mask_np] = 0' 로직을 삭제했습니다.
        # 이제 변수 'image'는 배경이 지워지지 않은 순수 원본 상태입니다.
        # ---------------------

        # 3. CLIP 임베딩 추출
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
            # L2 Normalization: $$ v = \frac{v}{\|v\|_2} $$
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            embedding = image_features.cpu().numpy().flatten()

        # 4. 결과 저장
        # entry에는 이미 'image_path', 'mask_path', 'is_crack', 'bbox'가 들어있습니다.
        entry['embedding'] = embedding.tolist() 
        results.append(entry)

    # 최종 메타데이터 저장
    output_path = os.path.join(root_path, 'metadata_with_clip_embeddings.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4) # 보기 좋게 indent 추가

    print(f"✅ Done! Embeddings saved to {output_path}")

generate_embeddings('/Dataset/khanhha_edit')