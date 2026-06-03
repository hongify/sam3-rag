import json
import numpy as np
import faiss
import torch
import os
import random
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = faiss.StandardGpuResources()

class RAGSearcher:
    def __init__(self, json_path, model_id="openai/clip-vit-large-patch14"):
        print("📥 Loading RAG Database (CLIP)...")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.db_data, self.gpu_idx = self.load_db_gpu(json_path)
        print("✅ RAG Database Loaded.")

    def load_db_gpu(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        crack_data = [d for d in data if d.get('is_crack', False)]
        
        embeddings = np.array([d['embedding'] for d in crack_data]).astype('float32')
        faiss.normalize_L2(embeddings)
        idx = faiss.IndexFlatIP(embeddings.shape[1])
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, idx)
        gpu_idx.add(embeddings)
        return crack_data, gpu_idx

    def get_image_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            q_vec = outputs.cpu().numpy().astype('float32')
            
        faiss.normalize_L2(q_vec)
        return q_vec

    def get_text_embedding(self, text_prompt):
        inputs = self.processor(text=[text_prompt], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            t_vec = outputs.cpu().numpy().astype('float32')
            
        faiss.normalize_L2(t_vec)
        return t_vec

    def crop_square_background(self, image_path, bbox_data, crop_size=224):
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            if w < crop_size or h < crop_size: return None
            bboxes = self.parse_bboxes_safe(bbox_data, w, h)
            
            for _ in range(30):
                rx = random.randint(0, w - crop_size)
                ry = random.randint(0, h - crop_size)
                crop_box = [rx, ry, rx + crop_size, ry + crop_size]
                
                is_safe = True
                for bbox in bboxes:
                    if self.check_overlap(crop_box, bbox, margin=100):
                        is_safe = False
                        break
                if is_safe:
                    return img.crop((rx, ry, rx + crop_size, ry + crop_size))

            for _ in range(20):
                rx = random.randint(0, w - crop_size)
                ry = random.randint(0, h - crop_size)
                crop_box = [rx, ry, rx + crop_size, ry + crop_size]
                
                is_safe = True
                for bbox in bboxes:
                    if self.check_overlap(crop_box, bbox, margin=50):
                        is_safe = False
                        break
                if is_safe:
                    return img.crop((rx, ry, rx + crop_size, ry + crop_size))
            return None 
        except:
            return None

    def search_exemplars(
        self, 
        query_path, 
        root_path, 
        text_prompt="A high contrast photo of a clear structural crack", 
        target_pos_count=2,
        target_neg_count=2, 
        seed=None,
        similarity_threshold=0.8 # 호환성을 위해 남겨둠 (내부 로직 변경)
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 1. 입력 이미지 벡터 추출
        img_vec = self.get_image_embedding(query_path)
        query_name = os.path.basename(query_path)
        
        # =========================================================
        # [Negative 추출] Image-to-Image (가장 유사한 배경 크롭)
        # =========================================================
        D_img, I_img = self.gpu_idx.search(img_vec, 200) # 200개 정도면 충분함
        neg_indices = list(zip(D_img[0], I_img[0]))
        # 💡 유사도가 가장 높은 상위 결과부터 쓰기 위해 shuffle 제거
        
        neg_res = []
        for d, i in neg_indices:
            if i < 0: continue 
            item = self.db_data[i]
            
            if os.path.basename(item['image_path']) == query_name: continue
            if d > 0.92: continue # 원본과 완전 동일한 이미지 배제

            full_path = item['image_path']
            if not os.path.isabs(full_path):
                full_path = os.path.join(root_path, os.path.basename(full_path))
            
            bg_img = self.crop_square_background(full_path, item.get('bbox', []), crop_size=224)
            if bg_img:
                new_item = item.copy()
                new_item['pil_image'] = bg_img 
                new_item['origin_full_path'] = full_path 
                new_item['rag_score'] = float(d) 
                neg_res.append(new_item)
            
            if len(neg_res) >= target_neg_count: break

        # =========================================================
        # [Positive 추출] Hybrid: Text(의미)로 1차 필터링 -> Image(시각)로 정렬
        # =========================================================
        text_vec = self.get_text_embedding(text_prompt)
        
        # 텍스트(VLM 묘사) 조건에 맞는 1차 후보군 200개 추출
        D_txt, I_txt = self.gpu_idx.search(text_vec, 200)
        
        pos_candidates = []
        for d, i in zip(D_txt[0], I_txt[0]):
            if i < 0: continue
            item = self.db_data[i]
            
            if os.path.basename(item['image_path']) == query_name: continue
            
            # 입력 이미지와의 시각적 유사도(질감/색상) 계산
            item_img_vec = np.array(item['embedding']).astype('float32')
            img_similarity = np.dot(img_vec.flatten(), item_img_vec.flatten())
            
            if img_similarity > 0.75: continue # 완전 동일한 이미지 배제
            
            # 💡 [핵심] 버리는 게 아니라, 이미지 유사도를 점수로 저장
            pos_candidates.append({
                'item': item,
                'img_sim': float(img_similarity),
                'text_sim': float(d)
            })

        # 💡 [핵심] 텍스트 조건을 만족하는 애들 중에서, 시각적 질감이 가장 높은 순으로 정렬!
        pos_candidates.sort(key=lambda x: x['img_sim'], reverse=True)

        pos_res = []
        for cand in pos_candidates:
            item = cand['item']
            full_path = item['image_path']
            if not os.path.isabs(full_path):
                full_path = os.path.join(root_path, os.path.basename(full_path))
            
            item['abs_path'] = full_path
            pos_res.append(item)
            
            if len(pos_res) >= target_pos_count: break

        return pos_res, neg_res

    # Utils
    def parse_bboxes_safe(self, bbox_data, w_img, h_img):
        if not bbox_data: return []
        try:
            arr = np.array(bbox_data)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            final_boxes = []
            for b in arr:
                if len(b) < 4: continue
                cx, cy, w, h = b[0], b[1], b[2], b[3]
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                final_boxes.append([x1, y1, x2, y2])
            return final_boxes
        except: return []

    def check_overlap(self, crop_box, crack_bbox, margin=50):
        cx1, cy1, cx2, cy2 = crop_box
        bx1 = crack_bbox[0] - margin
        by1 = crack_bbox[1] - margin
        bx2 = crack_bbox[2] + margin
        by2 = crack_bbox[3] + margin
        
        if cx2 < bx1 or cx1 > bx2 or cy2 < by1 or cy1 > by2:
            return False 
        return True