import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def process_metadata(root_path):
    # 경로 설정
    images_dir = os.path.join(root_path, 'images')
    masks_dir = os.path.join(root_path, 'masks')
    output_json = os.path.join(root_path, 'metadata.json')

    metadata = []
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"✅ Total images found: {len(image_files)}")

    for filename in tqdm(image_files):
        # 1. 경로 설정 (RAG에서 접근하기 쉽게 상대 경로로 저장)
        img_rel_path = os.path.join('./images', filename)
        
        # 마스크 파일 찾기 (이미지명과 동일하고 확장자만 다른 경우 대응)
        name_without_ext = os.path.splitext(filename)[0]
        mask_filename = name_without_ext + '.jpg' # 유저님의 데이터셋 기준 (.jpg)
        mask_full_path = os.path.join(masks_dir, mask_filename)
        mask_rel_path = os.path.join('./masks', mask_filename)

        is_crack = False
        bbox = [0, 0, 0, 0] # [cx, cy, w, h]
        
        # 2. 마스크 존재 여부 확인 및 BBox 추출
        if os.path.exists(mask_full_path):
            mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                # 마스크에서 픽셀값이 있는 부분 추출
                coords = cv2.findNonZero(mask)

                if coords is not None:
                    is_crack = True
                    # AABB (Axis-Aligned Bounding Box) 추출
                    x, y, w, h = cv2.boundingRect(coords)
                    
                    # [x, y, w, h] -> [cx, cy, w, h] 변환 (소수점 2자리)
                    cx = x + w / 2
                    cy = y + h / 2
                    bbox = [round(cx, 2), round(cy, 2), round(w, 2), round(h, 2)]
                else:
                    # 마스크 파일은 있지만 검은색뿐인 경우 (Negative 후보)
                    mask_rel_path = mask_rel_path 
            else:
                mask_rel_path = None # 읽기 실패 시
        else:
            mask_rel_path = None # 마스크 파일 자체가 없는 경우

        # 3. 데이터 저장 (mask_path 항목 추가)
        metadata.append({
            "image_path": img_rel_path,
            "mask_path": mask_rel_path, # 💡 핵심: 마스크 경로 추가
            "is_crack": is_crack,
            "bbox": bbox  # [cx, cy, w, h]
        })

    # JSON 저장
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"✨ Metadata saved successfully: {output_json}")

# 실행 예시
process_metadata('/Dataset/khanhha_edit')


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