# sam3_infer.py
import sys
import os

# --- Path Fix: 상위 폴더(SAM3 루트)를 sys.path에 추가 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.join(current_dir,"..","..",".."))
sys.path.append(parent_dir)
# -----------------------------------------------------

import torch
import traceback
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# SAM3 Imports
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.train.masks_ops import rle_encode
from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
from sam3.agent.viz import visualize

# GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_sam3_model(sam3_root_path):
    print("🏗️ Building SAM3 Model...")
    bpe_path = os.path.join(sam3_root_path, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.4)
    return processor

def convert_to_xyxy(box, img_w, img_h):
    """
    입력 박스가 Normalized인지 Pixel인지, CXCYWH인지 XYXY인지 판단하여
    항상 Pixel 단위 [x1, y1, x2, y2]로 반환합니다.
    Ref: Metadata BBox is [cx, cy, w, h] in Pixels.
    """
    if not box: return [0, 0, img_w, img_h]
    
    # 1. Normalized Check (<= 1.0) -> Assume Normalized CXCYWH
    if max(box) <= 1.0:
        cx, cy, w, h = box
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        return [int(x1), int(y1), int(x2), int(y2)]
    
    # 2. Absolute Check (> 1.0) -> Assume Pixel CXCYWH (from metadata)
    # Metadata bbox: [cx, cy, w, h]
    cx, cy, w, h = box
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return [int(x1), int(y1), int(x2), int(y2)]

def sam3_inference(processor, image_path, text_prompt, reference_infos=None):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")
        
    image = Image.open(image_path).convert("RGB")
    orig_img_w, orig_img_h = image.size
    inference_state = processor.set_image(image)
    activation_maps = [] 

    if reference_infos and isinstance(reference_infos, list):
        print(f"   Processing {len(reference_infos)} reference images...")
        for idx, ref_info in enumerate(reference_infos):
            r_path = ref_info.get("path")
            r_box = ref_info.get("box") # [cx, cy, w, h] (Pixel or Norm)
            ref_label = ref_info.get("label", True)
            
            if not r_path or not os.path.exists(r_path):
                print(f"   Skip invalid path: {r_path}")
                activation_maps.append(None)
                continue
            
            try:
                ref_image = Image.open(r_path).convert("RGB")
                ref_w, ref_h = ref_image.size
                
                # ★ 박스 좌표 변환 (Pixel XYXY로 통일)
                ref_box_pixel = convert_to_xyxy(r_box, ref_w, ref_h)

                result = processor.set_reference_prompt(
                    ref_image=ref_image, 
                    ref_box=ref_box_pixel,  # XYXY Pixel Expected
                    state=inference_state,
                    ref_label=ref_label,
                    return_heatmap=True
                )

                if isinstance(result, tuple) and len(result) == 2:
                    inference_state, heatmap = result
                    if heatmap is not None:
                        if not isinstance(heatmap, Image.Image):
                            heatmap = Sam3Processor.get_activation_heatmap(heatmap, (ref_w, ref_h))
                        activation_maps.append(heatmap)
                    else:
                        activation_maps.append(None)
                else:
                    inference_state = result
                    activation_maps.append(None)
                    
            except Exception as e:
                print(f"   Error on ref image {idx}: {e}")
                traceback.print_exc()
                activation_maps.append(None)

    final_prompt = text_prompt if text_prompt else "visual"
    predictions = processor.set_text_prompt(prompt=final_prompt, state=inference_state)

    # Format Output
    pred_boxes_xywh = []
    if predictions["boxes"].numel() > 0:
        pred_boxes_xyxy = torch.stack([
            predictions["boxes"][:, 0] / orig_img_w,
            predictions["boxes"][:, 1] / orig_img_h,
            predictions["boxes"][:, 2] / orig_img_w,
            predictions["boxes"][:, 3] / orig_img_h,
        ], dim=-1)
        pred_boxes_xywh = box_xyxy_to_xywh(pred_boxes_xyxy).tolist()
    
    pred_masks = []
    if "masks" in predictions and predictions["masks"] is not None:
        if predictions["masks"].numel() > 0:
            pred_masks_encoded = rle_encode(predictions["masks"].squeeze(1))
            pred_masks = [m["counts"] for m in pred_masks_encoded]

    outputs = {
        "orig_img_h": orig_img_h,
        "orig_img_w": orig_img_w,
        "pred_boxes": pred_boxes_xywh,
        "pred_masks": pred_masks,
        "pred_scores": predictions["scores"].tolist() if "scores" in predictions else [],
        "activation_maps": activation_maps
    }
    return outputs

def run_comprehensive_experiment(processor, image_path, text_prompt, reference_infos, output_folder_path="sam3_output"):
    print("\n🧪 [Experiment] Starting Generation...")
    
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    save_dir = os.path.join(output_folder_path, name_without_ext)
    os.makedirs(save_dir, exist_ok=True)

    scenarios = [
        ("1. Base (No Input)", None, None),
        ("2. Text Only", text_prompt, None),
        ("3. Ref Only", None, reference_infos),
        ("4. Text + Ref", text_prompt, reference_infos) 
    ]
    
    result_images = []
    scenario_activations = {}

    for name, t_prompt, r_infos in scenarios:
        try:
            resp = sam3_inference(processor, image_path, t_prompt, r_infos)
            resp["original_image_path"] = image_path
            
            viz = None
            if resp.get("pred_masks") and len(resp["pred_masks"]) > 0:
                try:
                    resp = remove_overlapping_masks(resp)
                    viz = visualize(resp)
                except Exception as viz_err:
                    print(f"   ⚠️ Viz skipped for {name}: {viz_err}")
                    viz = None 

            if viz is None: viz = Image.open(image_path).convert("RGB")
            result_images.append((name, viz))
            
            if "activation_maps" in resp:
                scenario_activations[name] = resp["activation_maps"]
            
        except Exception as e:
            print(f"   ❌ Critical Error in {name}: {e}")
            fail_img = Image.new('RGB', (512, 512), 'black')
            result_images.append((name + " (Fail)", fail_img))

    # --- Visualization Grid ---
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    
    def draw_box(img, box, color):
        cp = img.copy()
        draw = ImageDraw.Draw(cp)
        W, H = cp.size
        # 시각화 할 때도 동일한 변환 로직 사용
        x1, y1, x2, y2 = convert_to_xyxy(box, W, H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
        return cp

    # Row 1: References
    for i in range(4):
        ax = axes[0, i]; ax.set_axis_off()
        if reference_infos and i < len(reference_infos):
            ref = reference_infos[i]
            is_pos = ref.get('label', True)
            color = '#00FF00' if is_pos else '#FF0000'
            path = ref.get('path')
            
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB")
                
                # ★ 박스 그리기 추가
                if ref.get('box'):
                    img = draw_box(img, ref.get('box'), color)

                ax.imshow(img)
                ax.set_title(f"Ref {i+1} ({'POS' if is_pos else 'NEG'})", color=color, fontsize=16, fontweight='bold')
            else:
                ax.set_title("File Not Found")
        else:
            if i==0: ax.text(0.5,0.5,"No Ref", ha='center')
            ax.set_title("-")

    # Row 2: Activation Maps (Scenario 4)
    act_maps = scenario_activations.get("4. Text + Ref", [])
    for i in range(4):
        ax = axes[1, i]; ax.set_axis_off()
        if reference_infos and i < len(reference_infos):
            path = reference_infos[i].get('path')
            if path and os.path.exists(path):
                orig = Image.open(path).convert("RGB")
                
                # 원본에도 박스 표시
                ref_box = reference_infos[i].get('box')
                is_pos = reference_infos[i].get('label', True)
                color = '#00FF00' if is_pos else '#FF0000'
                if ref_box: orig = draw_box(orig, ref_box, color)

                if i < len(act_maps) and act_maps[i] is not None:
                    hm = act_maps[i]
                    if hm.size != orig.size: hm = hm.resize(orig.size, Image.Resampling.BILINEAR)
                    show = Image.blend(orig, hm, 0.5)
                    ax.set_title(f"Ref {i+1} Activation", color='blue', fontsize=16, fontweight='bold')
                else:
                    show = orig
                    ax.set_title("No Activation")
                ax.imshow(show)

    # Row 3: Results
    for i, (name, img) in enumerate(result_images):
        ax = axes[2, i]; ax.set_axis_off()
        ax.imshow(img)
        ax.set_title(name, fontsize=18, fontweight='bold')

    grid_path = os.path.join(save_dir, "experiment_grid.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"🎨 Saved Grid: {grid_path}")
    return grid_path