# rag_search.py
import json
import numpy as np
import faiss
import torch
import os
import random
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = faiss.StandardGpuResources()

class RAGSearcher:
    def __init__(self, json_path):
        print("📥 Loading RAG Database...")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
        self.db_data, self.gpu_idx = self.load_db_gpu(json_path)
        print("✅ RAG Database Loaded.")

    def load_db_gpu(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Crack이 있는 데이터만 필터링
        crack_data = [d for d in data if d.get('is_crack', False)]
        
        embeddings = np.array([d['embedding'] for d in crack_data]).astype('float32')
        faiss.normalize_L2(embeddings)
        idx = faiss.IndexFlatIP(embeddings.shape[1])
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, idx)
        gpu_idx.add(embeddings)
        return crack_data, gpu_idx

    def get_query_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            mid_layer = 12  # Large 기준 중간 layer
            patch_tokens = outputs.hidden_states[mid_layer][:, 1:, :]  # CLS 제외
            q_vec = patch_tokens.mean(dim=1).cpu().numpy().astype('float32')

        faiss.normalize_L2(q_vec)
        return q_vec

    def crop_square_background(self, image_path, bbox_data, crop_size=224):
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            if w < crop_size or h < crop_size: return None
            bboxes = self.parse_bboxes_safe(bbox_data, w, h)
            
            # 시도 1: Margin 100px (엄격)
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

            # 시도 2: Margin 50px (완화)
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

    def search_random(self, query_path, root_path, target_count=2, seed=None):
        """
        Negative Pool: 쿼리 벡터 그대로 검색 -> 가장 유사한 1000개
        Positive Pool: 쿼리 벡터에 (-) 붙여서 검색 -> 가장 안 비슷한 1000개
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        q_vec = self.get_query_embedding(query_path)
        query_name = os.path.basename(query_path)
        
        D_high, I_high = self.gpu_idx.search(q_vec, 1000)
        
        neg_indices = list(zip(D_high[0], I_high[0]))
        random.shuffle(neg_indices)


        D_low, I_low = self.gpu_idx.search(-q_vec, 1000)

        pos_indices = list(zip(D_low[0], I_low[0]))
        random.shuffle(pos_indices) # 랜덤 섞기

        neg_res = []
        for d, i in neg_indices:
            if i < 0: continue 
            item = self.db_data[i]
            
            if os.path.basename(item['image_path']) == query_name: continue
            if d > 0.9999: continue 

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
            
            if len(neg_res) >= target_count: break

        pos_res = []
        for d, i in pos_indices:
            if i < 0: continue
            item = self.db_data[i]
            
            if os.path.basename(item['image_path']) == query_name: continue
            
            full_path = item['image_path']
            if not os.path.isabs(full_path):
                full_path = os.path.join(root_path, os.path.basename(full_path))
            
            item['abs_path'] = full_path
            

            pos_res.append(item)
            
            if len(pos_res) >= target_count: break

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