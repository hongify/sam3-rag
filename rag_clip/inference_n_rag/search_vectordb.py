import json
import numpy as np
import faiss
import torch
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# --- 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = faiss.StandardGpuResources() 
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

# --- 1. 유틸리티 함수들 (BBox, Crop, DB Load) ---
# (이전 단계에서 검증된 코드 그대로 사용)

def parse_bboxes_safe(bbox_data, w_img, h_img):
    if not bbox_data: return []
    try:
        arr = np.array(bbox_data)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        final_boxes = []
        for b in arr:
            if len(b) < 4: continue
            b = list(b)
            if all(0 <= x <= 1.05 for x in b):
                b = [b[0]*w_img, b[1]*h_img, b[2]*w_img, b[3]*h_img]
            x1, y1 = b[0], b[1]
            if b[2] < x1: x2, y2 = x1 + b[2], y1 + b[3]
            else: x2, y2 = b[2], b[3]
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
            final_boxes.append([x1, y1, x2, y2])
        return final_boxes
    except:
        return []

def check_overlap(crop_box, crack_bbox, margin=20):
    cx1, cy1, cx2, cy2 = crop_box
    bx1, by1, bx2, by2 = crack_bbox[0]-margin, crack_bbox[1]-margin, crack_bbox[2]+margin, crack_bbox[3]+margin
    if cx2 < bx1 or cx1 > bx2 or cy2 < by1 or cy1 > by2: return False
    return True

def crop_square_background(image_path, bbox_data, crop_size=224):
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w < crop_size or h < crop_size: return None
        bboxes = parse_bboxes_safe(bbox_data, w, h)
        for _ in range(50):
            rx = random.randint(0, w - crop_size)
            ry = random.randint(0, h - crop_size)
            crop_box = [rx, ry, rx + crop_size, ry + crop_size]
            is_safe = True
            for bbox in bboxes:
                if check_overlap(crop_box, bbox, margin=20):
                    is_safe = False
                    break
            if is_safe:
                return img.crop((rx, ry, rx + crop_size, ry + crop_size))
        return None
    except:
        return None

def load_db_gpu(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    crack_data = [d for d in data if d['is_crack']]
    embeddings = np.array([d['embedding'] for d in crack_data]).astype('float32')
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    gpu_idx = faiss.index_cpu_to_gpu(res, 0, idx)
    gpu_idx.add(embeddings)
    return crack_data, gpu_idx

# --- 2. [수정됨] 랜덤 시드를 지원하는 검색 함수 ---

def search_rag_random(query_path, db_data, gpu_idx, root_path, target_count=2, seed=None):
    """
    seed: 정수값. None이면 완전 랜덤, 값이 있으면 고정된 랜덤 결과 반환.
    """
    
    # 시드 설정
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    img = Image.open(query_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        q_vec = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
    faiss.normalize_L2(q_vec)
    
    query_name = os.path.basename(query_path)
    print(f"Query: {query_name} (Seed: {seed})")

    # 한번에 500개를 검색 (정방향 벡터 사용)
    # 상위권 -> Negative (배경용, 아주 비슷함)
    # 중위권 -> Positive (데이터용, 비슷하지만 다름)
    D, I = gpu_idx.search(q_vec, 2000) 
    all_candidates = list(zip(D[0], I[0]))

    # =========================================================
    # 1. Negative 찾기 (Background) - 상위권(Top 50) 활용
    # =========================================================
    neg_res = []
    
    # 상위 50개 후보군 추출
    neg_candidates = all_candidates[:50]
    
    # 랜덤성을 위해 섞음
    random.shuffle(neg_candidates)
    
    for d, i in neg_candidates:
        item = db_data[i]
        item_name = os.path.basename(item['image_path'])
        
        # 자기 자신 제외
        if item_name == query_name: continue
        # 완전 복제본 수준 제외
        if d > 0.999: continue

        full_path = item['image_path']
        if not os.path.isabs(full_path):
            full_path = os.path.join(root_path, item_name)
            
        bg_img = crop_square_background(full_path, item.get('bbox', []), crop_size=224)
        
        if bg_img:
            new_item = item.copy()
            new_item['pil_image'] = bg_img
            neg_res.append(new_item)
            
        if len(neg_res) >= target_count: break
            
    print(f"   Negative: {len(neg_res)} picked from Top-50 (High Similarity)")

    # =========================================================
    # 2. Positive 찾기 (Similar but Distinct) - 중위권(Rank 50~300) 활용
    # =========================================================
    pos_res = []
    
    # 50등부터 300등 사이의 후보군 추출 (너무 똑같지도, 너무 다르지도 않은 구간)
    pos_pool = []
    for d, i in all_candidates[-30:-10]:
        item = db_data[i]
        if os.path.basename(item['image_path']) == query_name: continue
        pos_pool.append(item)
    
    # 후보군 중에서 랜덤 샘플링
    if len(pos_pool) >= target_count:
        pos_res = random.sample(pos_pool, target_count)
    else:
        pos_res = pos_pool # 부족하면 있는 대로 다 가져옴
        
    print(f"   Positive: {len(pos_res)} picked from Rank 50-300 (Mid Similarity)")

    return pos_res, neg_res

# --- 3. 시각화 (경로 수정 포함) ---
def visualize(query_path, pos, neg, root_path, output_name):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    try: axes[0].imshow(Image.open(query_path))
    except: pass
    axes[0].set_title("Query Image", fontweight='bold')
    
    for i, p in enumerate(pos[:2]):
        p_path = p['image_path']
        if not os.path.isabs(p_path):
            p_path = os.path.join(root_path, os.path.basename(p_path))
        try:
            axes[1+i].imshow(Image.open(p_path))
            axes[1+i].set_title(f"POS {i+1}\n(Random Pick)", color='green')
        except: pass
            
    for i, n in enumerate(neg[:2]):
        axes[4+i].imshow(n['pil_image'])
        axes[4+i].set_title(f"NEG {i+1}\n(Random Crop)", color='red')
        
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_name)
    print(f"✅ Saved: {output_name}")

# --- 실행 ---
if __name__ == "__main__":
    json_path = '/Dataset/khanhha_edit/metadata_with_embeddings.json'
    img_root = '/Dataset/khanhha_edit/images'
    test_img = '/workspace/SAM3RAG/assets/cracks/GAPS384_train_0685_541_641.jpg'
    
    db, idx = load_db_gpu(json_path)
    
    # # 1. 시드 고정 (항상 같은 랜덤 결과) -> 실험 재현용
    # print("\n--- Try 1: Seed 42 (Fixed) ---")
    # p1, n1 = search_rag_random(test_img, db, idx, img_root, seed=42)
    # visualize(test_img, p1, n1, img_root, "rag_seed_42.png")
    
    # 2. 시드 없음 (매번 다른 결과) -> 실제 서비스용
    print("\n--- Try 2: No Seed (Random) ---")
    p2, n2 = search_rag_random(test_img, db, idx, img_root, seed=None)
    visualize(test_img, p2, n2, img_root, "rag_random.png")