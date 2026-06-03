import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/dinov2-large"

# 1. 프로세서 설정 (warning 방지를 위해 use_fast=True 권장)
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

# 2. 모델 설정: attn_implementation="eager"가 핵심입니다.
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation="eager"
).to(device)

def get_single_image_attention(image_path, target_layers=[4,8,12,16,20,23]):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # output_attentions=True를 명시적으로 설정
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    
    attentions = outputs.attentions 
    
    if attentions is None:
        print("Error: Attentions are still None. Check model implementation.")
        return

    # 시각화 부분
    fig, axes = plt.subplots(1, len(target_layers) + 1, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # DINOv2-Large의 패치 그리드 계산 (Patch size 14)
    w, h = inputs['pixel_values'].shape[-2:]
    grid_w, grid_h = w // 14, h // 14

    for i, layer_idx in enumerate(target_layers):
        attn = attentions[layer_idx][0] 
        avg_attn = attn.mean(dim=0)     
        cls_attn = avg_attn[0, 1:]      
        
        # 1. 1D 데이터를 2D 그리드로 변환 (예: 16x16)
        mask = cls_attn.reshape(1, 1, grid_h, grid_w)
        
        # 2. Bilinear Interpolation으로 원본 해상도(w, h)까지 확대
        # 이 과정에서 뭉툭한 격자가 부드러운 히트맵으로 변합니다.
        mask = F.interpolate(mask, size=(w, h), mode='bilinear', align_corners=False)
        mask = mask.squeeze().cpu().numpy()
        
        # 정규화
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        
        # 원본 이미지 위에 겹쳐서 그리기 (Alpha값 조절)
        axes[i+1].imshow(image) # 배경에 원본 이미지
        axes[i+1].imshow(mask, cmap='jet', alpha=0.5) # 그 위에 히트맵 투명하게 덮기
        axes[i+1].set_title(f"Layer {layer_idx} Overlay")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig('attention_map_result.png') # 서버 환경일 수 있으니 저장 추가
    plt.show()

    # CLS 토큰 임베딩 반환
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

# 실행 (이미지 파일명이 정확한지 확인하세요)
embedding = get_single_image_attention('DJ_Wall_199.jpg')