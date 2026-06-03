import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import random

def process_metadata(root_path):
    images_dir = os.path.join(root_path, 'images')
    masks_dir = os.path.join(root_path, 'masks')
    output_json = os.path.join(root_path, 'metadata.json')

    metadata = []
    
    # 이미지 파일 목록 가져오기 (확장자 대응)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Total images found: {len(image_files)}")

    for filename in tqdm(image_files):
        img_path = os.path.join('./images', filename)
        # 마스크 파일은 이미지와 이름이 같고 확장자가 .png라고 가정 (필요시 수정)
        mask_name = os.path.splitext(filename)[0] + '.jpg'
        mask_path = os.path.join(masks_dir, mask_name)

        is_crack = False
        bbox = [0, 0, 0, 0] # cx, cy, w, h

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            coords = cv2.findNonZero(mask)

            if coords is not None:
                is_crack = True
                # 좌표로부터 최소 사각형(AABB) 추출
                x, y, w, h = cv2.boundingRect(coords)
                
                # (x, y, w, h) -> (cx, cy, w, h) 변환
                cx = x + w / 2
                cy = y + h / 2
                bbox = [round(cx, 2), round(cy, 2), round(w, 2), round(h, 2)]

        # 데이터 저장
        metadata.append({
            "image_path": img_path,
            "is_crack": is_crack,
            "bbox": bbox  # [cx, cy, w, h]
        })

    # JSON 저장
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {output_json}")

# 실행
# process_metadata('/Dataset/khanhha_edit')


def save_verification_image(root_path, output_filename='verification_result.png'):
    json_path = os.path.join(root_path, 'metadata.json')
    
    # 1. JSON 로드
    if not os.path.exists(json_path):
        print(f"Error: {json_path}를 찾을 수 없습니다.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 2. 균열이 있는 샘플 우선 선택 (검증을 위해)
    crack_samples = [m for m in metadata if m['is_crack']]
    sample = random.choice(crack_samples) if crack_samples else random.choice(metadata)
    
    # 3. 이미지 로드
    # metadata.json의 image_path가 './images/...' 형태이므로 root_path와 결합
    relative_path = sample['image_path'].lstrip('./') 
    img_full_path = os.path.join(root_path, relative_path)
    
    image = cv2.imread(img_full_path)
    if image is None:
        print(f"Error: 이미지를 불러올 수 없습니다. 경로를 확인하세요: {img_full_path}")
        return
    
    # 4. BBox 그리기 (is_crack이 True일 때만)
    if sample['is_crack']:
        cx, cy, w, h = sample['bbox']
        
        # cx, cy, w, h -> x1, y1, x2, y2 변환
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        # 초록색 박스 (BGR: 0, 255, 0), 두께 3
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 텍스트 추가 (선택 사항)
        cv2.putText(image, "Crack Detected", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Crack", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 5. 결과 저장
    cv2.imwrite(output_filename, image)
    print(f"검증 이미지가 저장되었습니다: {os.path.abspath(output_filename)}")

# 실행
# save_verification_image('/Dataset/khanhha_edit')