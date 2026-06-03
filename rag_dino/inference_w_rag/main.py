# main.py
import os
import shutil
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import sam3_infer
from rag_search import RAGSearcher

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.join(current_dir,"..","..",".."))
sys.path.append(parent_dir)
SAM3_ROOT = parent_dir 

CONFIG = {
    "json_path": '/Dataset/khanhha_edit/metadata_with_dino_embeddings.json',
    "img_root": '/Dataset/khanhha_edit/images',
    "temp_dir": "./temp_refs", 
    "output_dir": "./results"
}

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["temp_dir"], exist_ok=True)
    
    rag = RAGSearcher(CONFIG["json_path"])
    sam_processor = sam3_infer.load_sam3_model(SAM3_ROOT)

    target_img = f"{SAM3_ROOT}/assets/cracks/GAPS384_train_0541_1_641.jpg"
    text_input = "crack"

    print(f"\n🚀 Start Pipeline for: {os.path.basename(target_img)}")

    pos_items, neg_items = rag.search_random(
        target_img, 
        CONFIG["img_root"], 
        target_count=2, 
        seed=None
    )

    ref_list = []

    for i, item in enumerate(pos_items):
        bbox = item.get('bbox', [0.5, 0.5, 1.0, 1.0])
        
        ref_list.append({
            "path": item['abs_path'], 
            "box": bbox, # Pixel Coordinates: [cx, cy, w, h]
            "label": True # Positive
        })
        print(f"   [+] Pos Ref {i+1}: {os.path.basename(item['abs_path'])} | Box: {bbox}")

    for i, item in enumerate(neg_items):
        temp_path = os.path.join(CONFIG["temp_dir"], f"neg_crop_{i}.jpg")
        item['pil_image'].save(temp_path)
        
        ref_list.append({
            "path": temp_path,
            "box": [0.5, 0.5, 1.0, 1.0], 
            "label": False 
        })
        print(f"   [-] Neg Ref {i+1}: Saved to {temp_path} | Box: [Full Image]")

    sam3_infer.run_comprehensive_experiment(
        sam_processor,
        image_path=target_img,
        text_prompt=text_input,
        reference_infos=ref_list,
        output_folder_path=CONFIG["output_dir"]
    )


if __name__ == "__main__":
    main()