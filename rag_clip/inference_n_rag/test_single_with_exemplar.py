# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import json
import os
import torch
import traceback
import numpy as np
from PIL import Image, ImageDraw

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xyxy_to_xywh, box_cxcywh_to_xyxy
from sam3.train.masks_ops import rle_encode
from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
from sam3.agent.viz import visualize
import matplotlib.pyplot as plt

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 경로 설정
import sam3
sam3_root = os.path.dirname(os.path.abspath(__file__)) # 혹은 실제 루트 경로
# sam3_root = "/workspace/SAM3RAG" # 필요시 절대 경로로 수정하세요
bpe_path = os.path.join(sam3_root, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")

# 모델 빌드
print(" Building SAM3 Model...")
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.4) # Threshold 조정 가능


def sam3_inference(processor, image_path, text_prompt,
                   reference_infos=None,  # List[Dict]
                   debug_save_prefix=None): # 하위 호환성을 위해 인자는 유지하나 사용 안 함
    """Run SAM 3 image inference and return activation maps in memory."""
    
    # 1. 타겟 이미지 로드
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")
        
    image = Image.open(image_path).convert("RGB")
    orig_img_w, orig_img_h = image.size
    inference_state = processor.set_image(image)
    
    activation_maps = [] # [Image, Image, ...]

    if reference_infos and isinstance(reference_infos, list):
        print(f" Processing {len(reference_infos)} reference images...")
        
        for idx, ref_info in enumerate(reference_infos):
            # 정보 추출
            r_path = ref_info.get("path")
            r_box = ref_info.get("box") # Normalized Box or Absolute
            ref_label = ref_info.get("label", True)
            
            if not r_path or not os.path.exists(r_path):
                print(f" Skip invalid path: {r_path}")
                activation_maps.append(None) # 인덱스 유지를 위해 None 추가
                continue
                
            try:
                # 이미지 로드
                ref_image = Image.open(r_path).convert("RGB")
                ref_w, ref_h = ref_image.size
                
                # 좌표 변환 (Normalized -> Pixel)
                if max(r_box) <= 1.0:
                    cx, cy, w, h = r_box
                    x1, y1, x2, y2 = (cx-w/2)*ref_w, (cy-h/2)*ref_h, (cx+w/2)*ref_w, (cy+h/2)*ref_h
                    ref_box_pixel = [int(x1), int(y1), int(x2), int(y2)]
                else:
                    ref_box_pixel = r_box

                # ★ 중요: processor가 (state, heatmap)을 반환하도록 수정되었다고 가정합니다.
                # 만약 processor 코드를 수정할 수 없다면, 이 부분 로직 확인이 필요합니다.
                result = processor.set_reference_prompt(
                    ref_image=ref_image, 
                    ref_box=ref_box_pixel, 
                    state=inference_state,
                    ref_label=ref_label,
                    return_heatmap=True  )

                if isinstance(result, tuple) and len(result) == 2:
                    inference_state, heatmap = result
                    
                    if heatmap is not None:
                        # 혹시 Tensor나 Array로 왔다면 변환
                        if not isinstance(heatmap, Image.Image):
                            heatmap = Sam3Processor.get_activation_heatmap(heatmap, (ref_w, ref_h))
                        
                        # 변환된(혹은 원래 있던) 이미지를 리스트에 추가
                        activation_maps.append(heatmap)
                    else:
                        activation_maps.append(None)
                else:
                    inference_state = result
                    activation_maps.append(None)
                    
                print(f"   Assertion {idx+1}: {os.path.basename(r_path)} processed.")
                
            except Exception as e:
                print(f" Error on ref image {idx}: {e}")
                traceback.print_exc()
                activation_maps.append(None)

    # 4. 텍스트 프롬프트 설정 및 인퍼런스 실행
    final_prompt = text_prompt if text_prompt else "visual"
    
    predictions = processor.set_text_prompt(
        prompt=final_prompt, 
        state=inference_state
    )

    # 5. 결과 포맷팅
    if predictions["boxes"].numel() > 0:
        pred_boxes_xyxy = torch.stack(
            [
                predictions["boxes"][:, 0] / orig_img_w,
                predictions["boxes"][:, 1] / orig_img_h,
                predictions["boxes"][:, 2] / orig_img_w,
                predictions["boxes"][:, 3] / orig_img_h,
            ],
            dim=-1,
        )
        pred_boxes_xywh = box_xyxy_to_xywh(pred_boxes_xyxy).tolist()
    else:
        pred_boxes_xywh = []
    
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
        "activation_maps": activation_maps # [New] 메모리에 저장된 히트맵 리스트 반환
    }
    
    return outputs


def call_sam_service(
    sam3_processor,
    image_path: str,
    text_prompt: str,
    output_folder_path: str = "sam3_output",
    reference_infos: list = None
):
    print(f"\n Processing: {os.path.basename(image_path)}")
    print(f"   Prompt: '{text_prompt}'")
    if reference_infos:
        for idx, ref_info in enumerate(reference_infos):
            print(f"[{idx}]   Ref: {os.path.basename(ref_info.get('path'))} Positive={ref_info.get('label')}")

    text_prompt_safe = text_prompt.replace("/", "_") if text_prompt else "no_text"
    
    save_dir = os.path.join(output_folder_path, image_path.replace("/", "-"))
    os.makedirs(save_dir, exist_ok=True)
    
    output_json_path = os.path.join(save_dir, f"{text_prompt_safe}.json")
    output_image_path = os.path.join(save_dir, f"{text_prompt_safe}.png")
    activation_save_prefix = os.path.join(save_dir, f"{text_prompt_safe}_ref_activation.png")

    try:
        # [수정] inference 호출 시 경로 전달
        serialized_response = sam3_inference(
            sam3_processor, 
            image_path, 
            text_prompt,
            reference_infos=reference_infos, # 리스트 전달
            debug_save_prefix=activation_save_prefix 
        )

        # 후처리 (중복 제거 등)
        serialized_response = remove_overlapping_masks(serialized_response)
        serialized_response.update({
            "original_image_path": image_path,
            "output_image_path": output_image_path,
        })

        # 점수순 정렬
        if "pred_scores" in serialized_response and serialized_response["pred_scores"]:
            indices = sorted(
                range(len(serialized_response["pred_scores"])),
                key=lambda i: serialized_response["pred_scores"][i],
                reverse=True,
            )
            for key in ["pred_scores", "pred_boxes", "pred_masks"]:
                serialized_response[key] = [serialized_response[key][i] for i in indices]

        # JSON 저장
        with open(output_json_path, "w") as f:
            json.dump(serialized_response, f, indent=4)
        print(f" JSON Saved: {output_json_path}")

        # 시각화 및 이미지 저장
        print(" Rendering Visualization...")
        # visualize 함수가 내부적으로 이미지를 로드하므로 경로가 정확해야 함
        viz_image = visualize(serialized_response) 
        viz_image.save(output_image_path)
        print(f" Image Saved: {output_image_path}")

    except Exception as e:
        print(f" Error: {e}")
        traceback.print_exc()

    return output_json_path

#===============
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

def run_comprehensive_experiment(processor, image_path, text_prompt, reference_infos, output_folder_path="sam3_output"):
    """
    [3행 구성 고해상도 리포트 - In-Memory 처리 버전]
    중간 파일 저장 없이, 추론 결과(resp)에서 이미지를 바로 꺼내 Grid를 생성합니다.
    """
    print("\n🧪 [Experiment] Starting Generation (Memory Mode)...")
    
    # 1. 저장 경로 설정 (폴더만 생성)
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    save_dir = os.path.join(output_folder_path, name_without_ext)
    os.makedirs(save_dir, exist_ok=True)

    # 2. 시나리오 정의
    # (prefix 인자는 이제 파일 저장용이 아니라 디버깅용으로 쓰거나 무시됩니다)
    scenarios = [
        ("1. Base (No Input)", None, None),
        ("2. Text Only", text_prompt, None),
        ("3. Ref Only", None, reference_infos),
        ("4. Text + Ref", text_prompt, reference_infos) 
    ]
    
    result_images = []
    scenario_activations = {} # 각 시나리오별 Activation Map 저장소

    # 3. 추론 실행
    print("   Running inference for scenarios...")
    for name, t_prompt, r_infos in scenarios:
        try:
            # 1. 추론 실행
            resp = sam3_inference(processor, image_path, t_prompt, r_infos)
            resp["original_image_path"] = image_path
            
            viz = None # 초기화

            # 2. 마스크 시각화 시도 (안전장치 추가)
            if resp.get("pred_masks") and len(resp["pred_masks"]) > 0:
                try:
                    # ★ 여기서 에러(zero-size array)가 발생하면 catch 블록으로 이동
                    resp = remove_overlapping_masks(resp)
                    viz = visualize(resp)
                except Exception as viz_err:
                    # 마스크 처리 중 에러가 나면 그냥 '탐지 실패'로 간주하고 로그만 남김
                    print(f"   ⚠️ Visualization skipped for {name} (Empty mask?): {viz_err}")
                    viz = None 

            # 3. 시각화 실패했거나 마스크가 없으면 -> 원본 이미지 사용
            if viz is None:
                viz = Image.open(image_path).convert("RGB")

            # 4. 결과 저장
            result_images.append((name, viz))
            
            # Activation Map 메모리에 보관
            if "activation_maps" in resp:
                scenario_activations[name] = resp["activation_maps"]
            
        except Exception as e:
            # 여기는 추론 자체가 터졌을 때 (치명적 에러)
            print(f"   ❌ Critical Error in {name}: {e}")
            fail_img = Image.new('RGB', (512, 512), 'black')
            result_images.append((name + " (Fail)", fail_img))

    # 4. 그리드 그리기 (3 Rows x 4 Cols)
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    # fig.suptitle(f"SAM3 Analysis: {base_name}", fontsize=24, fontweight='bold', y=0.95)

    # 공통 함수: 박스 그리기
    def draw_box_on_image(img, box, color, width=6):
        # 원본을 훼손하지 않기 위해 복사
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        W, H = img_copy.size
        if max(box) <= 1.0: # Normalized
            cx, cy, w, h = box
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
        else: # Absolute
            x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        return img_copy

    # ==========================================
    # Row 1: Original Reference Images
    # ==========================================
    for i in range(4):
        ax = axes[0, i]
        ax.set_axis_off()
        
        if reference_infos and i < len(reference_infos):
            ref = reference_infos[i]
            orig_ref_path = ref.get('path')
            is_pos = ref.get('label', True)
            box_color = '#00FF00' if is_pos else '#FF0000'
            
            if orig_ref_path and os.path.exists(orig_ref_path):
                img = Image.open(orig_ref_path).convert("RGB")
                # if ref.get('box'):
                #     img = draw_box_on_image(img, ref.get('box'), box_color)
                
                ax.imshow(img)
                ax.set_title(f"Ref {i+1} Original\n({'POS' if is_pos else 'NEG'})", 
                             color=box_color, fontsize=16, fontweight='bold')
            else:
                ax.set_title("Image Not Found", fontsize=14)
                ax.imshow(np.full((100, 100, 3), 50, dtype=np.uint8))
        else:
            if i == 0: ax.text(0.5, 0.5, "No References", ha='center')
            ax.imshow(np.full((100, 100, 3), 240, dtype=np.uint8))
            ax.set_title("-", fontsize=14)

    # ==========================================
    # Row 2: Activation Maps (From Memory)
    # ==========================================
    # "4. Text + Ref" 시나리오의 Activation Map을 사용합니다.
    target_scenario_name = "4. Text + Ref"
    act_maps = scenario_activations.get(target_scenario_name, [])

    for i in range(4):
        ax = axes[1, i]
        ax.set_axis_off()
        
        if reference_infos and i < len(reference_infos):
            ref = reference_infos[i]
            is_pos = ref.get('label', True)
            box_color = '#00FF00' if is_pos else '#FF0000'
            orig_ref_path = ref.get('path')
            
            img_to_show = None
            title_text = ""

            if orig_ref_path and os.path.exists(orig_ref_path):
                orig_img = Image.open(orig_ref_path).convert("RGB")
                
                # [수정된 부분] 인덱스 체크 및 None 체크 강화
                has_heatmap = False
                if i < len(act_maps):
                    heatmap_img = act_maps[i]
                    if heatmap_img is not None: # <--- 여기서 None인지 확실히 체크!
                        has_heatmap = True
                        # 사이즈 체크 및 리사이즈
                        if heatmap_img.size != orig_img.size:
                            heatmap_img = heatmap_img.resize(orig_img.size, Image.Resampling.BILINEAR)
                        
                        img_to_show = Image.blend(orig_img, heatmap_img, alpha=0.5)
                        title_text = f"Ref {i+1} Activation"
                
                if not has_heatmap:
                    img_to_show = orig_img
                    title_text = "No Activation Map"
                
                # 박스 그리기
                if ref.get('box'):
                    img_to_show = draw_box_on_image(img_to_show, ref.get('box'), box_color)

                ax.imshow(img_to_show)
                ax.set_title(title_text, color=box_color, fontsize=16, fontweight='bold')
            else:
                ax.imshow(np.full((100, 100, 3), 50, dtype=np.uint8))
                ax.set_title("Ref Img Missing", fontsize=12)
        else:
            ax.imshow(np.full((100, 100, 3), 240, dtype=np.uint8))

    # ==========================================
    # Row 3: Scenario Results
    # ==========================================
    for i, (name, img) in enumerate(result_images):
        ax = axes[2, i]
        ax.set_axis_off()
        ax.imshow(img)
        ax.set_title(name, fontsize=18, fontweight='bold')

    # 5. 최종 저장 (High DPI)
    grid_save_path = os.path.join(save_dir, "experiment_grid_final.png")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.2, wspace=0.1) 
    plt.savefig(grid_save_path, dpi=300, bbox_inches='tight') 
    plt.close()
    
    print(f"🎨 Final Grid Saved (No Temp Files): {grid_save_path}")
    return grid_save_path

# ==========================================
# 실행부 (Main Execution)



#workspace/SAM3RAG/assets/cracks/cc16b783c623d23a8a003e951ca12e65510b2e0b.jpeg
# DJ_Wall_199.jpg
target_img = f"{sam3_root}/assets/cracks/000087.jpg" # 찾고 싶은 대상이 있는 이미지


text_input = "crack" 
ref_list = None



ref_list = [
    {
        "path": f"{sam3_root}/assets/cracks/Volker_DSC01608_737_269_1058_1389.jpg",
        "box": [0.5,0.5,1.0,1.0],  
        "label": True 
    },
            {
        "path": f"{sam3_root}/assets/cracks/CFD_025.jpg", 
        "box": [0.5,0.5,1.0,1.0],   
        "label": True
    },
    {
        "path": f"{sam3_root}/assets/cracks/000871.jpg", 
        "box": [0.75, 0.5, 0.3, 1.0],  
        "label": False
    },
            {
        "path": f"{sam3_root}/assets/cracks/000089.jpg",
        "box": [0.25, 0.75, 0.5, 0.5],  
        "label": False 
    },
]




if __name__ == "__main__":
    # call_sam_service(
    #     processor, 
    #     image_path=target_img, 
    #     text_prompt=text_input, 
    #     output_folder_path="./test_sam3_output_result",
    #     reference_infos=ref_list
    # )
    run_comprehensive_experiment(
        processor,
        image_path=target_img,
        text_prompt=text_input,
        reference_infos=ref_list,
        output_folder_path="./test_sam3_output_result"
    )