import cv2
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_background_patches(
    image_dir, 
    mask_dir, 
    output_dir, 
    patch_size=(224, 224), 
    patches_per_image=2
):
    """
    이미지와 마스크를 읽어서, 크랙이 '전혀 없는' 배경 부분만 크롭하여 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 목록 가져오기
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    
    count = 0
    
    for img_path in tqdm(image_paths, desc="배경 패치 생성 중"):
        # 1. 이미지와 마스크 로드
        img_name = img_path.name
        mask_name = img_name # 마스크 이름 규칙에 맞게 수정 필요 (예: _mask.png 등)
        
        # 데이터셋마다 마스크 경로 규칙이 다를 수 있음 (예: png, jpg)
        mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + ".png") 
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, img_name) # 같은 확장자 시도
            
        if not os.path.exists(mask_path):
            continue # 마스크 없으면 스킵

        img = cv2.imread(str(img_path))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 0:배경, 255:크랙

        if img is None or mask is None: continue
        
        h, w, _ = img.shape
        ph, pw = patch_size

        # 이미지가 패치보다 작으면 스킵
        if h < ph or w < pw: continue

        # 2. 랜덤 크롭 시도 (최대 50번 시도)
        success_count = 0
        for _ in range(50): 
            if success_count >= patches_per_image: break
            
            # 랜덤 좌표 선정
            y = random.randint(0, h - ph)
            x = random.randint(0, w - pw)
            
            # 마스크 크롭 (해당 영역에 크랙이 있는지 확인)
            mask_crop = mask[y:y+ph, x:x+pw]
            
            # 3. 크랙 픽셀이 하나도 없으면(0이면) 저장 (노이즈 감안해서 sum < 10 등으로 조절 가능)
            if np.sum(mask_crop) == 0: 
                # 이미지 크롭
                img_crop = img[y:y+ph, x:x+pw]
                
                # 저장 (파일명: 원본명_bg_01.jpg)
                save_name = f"{os.path.splitext(img_name)[0]}_bg_{success_count}.jpg"
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, img_crop)
                
                success_count += 1
                count += 1

    print(f"✅ 완료! 총 {count}장의 배경(Non-crack) 이미지가 생성되었습니다.")
    print(f"저장 위치: {output_dir}")

# --- 실행 예시 ---
# image_folder = "/Dataset/khanhha_edit/images"
# mask_folder = "/Dataset/khanhha_edit/masks" # 마스크가 있는 폴더
# output_folder = "/Dataset/khanhha_edit/background_patches"

# generate_background_patches(image_folder, mask_folder, output_folder)