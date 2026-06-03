import os
import sys
import json
from PIL import Image

from google import genai
from google.genai import types

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.join(current_dir,"..","..",".."))
sys.path.append(parent_dir)

import sam3_infer
from rag_search import RAGSearcher

SAM3_ROOT = parent_dir 

CONFIG = {
    "json_path": '/Dataset/khanhha_edit/metadata_with_clip_embeddings.json', 
    "img_root": '/Dataset/khanhha_edit/images',
    "temp_dir": "./temp_refs", 
    "output_dir": "./results",
    "use_vlm": True, 
    "default_prompt": "A high contrast photo of a clear structural crack" 
}

def generate_search_prompt_with_vlm(image_path):

    print("[VLM] Analyzing target image to generate search strategy...")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        
    mime_type = 'image/png' if image_path.lower().endswith('.png') else 'image/jpeg'

    system_instruction = """
    You are an expert AI vision analyst for a Vision Foundation Model (SAM3).
    We need to find a 'Positive Exemplar' image from our database.
    
    CRITICAL RULE: The exemplar MUST match the exact visual characteristics of the crack in the provided image (e.g., thickness, faintness, continuity, contrast), but it SHOULD NOT focus on matching the background texture. 

    Step 1. Closely examine the exact nature of the crack: Is it a faint hairline, a deep structural gap, a blurred weathered line, or a sharp spiderweb? 
    Step 2. Ignore the specific background material (asphalt, concrete, dirt, etc.).
    Step 3. Write a highly descriptive CLIP search prompt that focuses strictly on the crack's specific visual state and geometry. Do NOT force "high-contrast" if the original crack is faint. Match the condition.

    Return ONLY the final search prompt string. No quotes, no markdown, no explanations.
    
    Example output (if the input is a very faint, thin crack):
    A subtle, faint hairline crack, low-contrast thin fracture line, barely visible surface fissure
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            system_instruction
        ]
    )
    
    generated_prompt = response.text.strip().replace('"', '')
    return generated_prompt

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["temp_dir"], exist_ok=True)
    
    rag = RAGSearcher(CONFIG["json_path"])
    sam_processor = sam3_infer.load_sam3_model(SAM3_ROOT)

    target_img = f"{SAM3_ROOT}/assets/cracks/GAPS384_train_0685_541_641.jpg"
    text_input = "crack"

    print(f"\n🚀 Start Pipeline for: {os.path.basename(target_img)}")

    if CONFIG["use_vlm"]:
        search_prompt = generate_search_prompt_with_vlm(target_img)
        print(f"💡 [Mode: VLM ON] Generated Prompt: '{search_prompt}'")
    else:
        search_prompt = CONFIG["default_prompt"]
        print(f"💡 [Mode: VLM OFF] Using Default Prompt: '{search_prompt}'")

    pos_items, neg_items = rag.search_exemplars(
        query_path=target_img, 
        root_path=CONFIG["img_root"], 
        text_prompt=search_prompt, 
        target_pos_count=6,
        target_neg_count=2, 
        seed=None,
        similarity_threshold=0.8 
    )

    ref_list = []

    for i, item in enumerate(pos_items):
        bbox = item.get('bbox', [0.5, 0.5, 1.0, 1.0])
        ref_list.append({
            "path": item['abs_path'], 
            "box": bbox, 
            "label": True 
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